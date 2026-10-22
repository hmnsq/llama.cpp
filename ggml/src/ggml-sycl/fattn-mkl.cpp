// Flash attention via oneMKL GEMM (XMX-accelerated).
// Uses column_major::gemm for Q*K^T and S*V matmuls
// with an online softmax SYCL kernel.
//
// All GQA query heads sharing a KV head are batched into single
// GEMM calls, amortizing MKL launch overhead across K and V reuse.
//

#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "fattn-common.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"
#ಅನ್ತರ್ಭಾವ "fattn.hpp"

#ಅನ್ತರ್ಭಾವ <oneapi/mkl.hpp>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <chrono>

#ವ್ಯಾಖ್ಯಾನಿಸು MKL_FA_CHUNK_ಗಾತ್ರ_KV 8192

// Number of query rows processed per tile. The score buffers (KQ_f32, S_f16)
// are sized q_tile_rows * chunk_size, so this bounds their footprint
// regardless of batch size (n_query_rows = n_queries * gqa_ratio). A typical
// single-ubatch prefill (e.g. ubatch 1024 * gqa 8 = 8192 rows) is exactly one
// tile, so it runs with no extra iterations. Larger batches tile and stay
// bounded. Override with GGML_SYCL_MKL_FA_Q_TILE.
#ವ್ಯಾಖ್ಯಾನಿಸು MKL_FA_Q_TILE 8192

#ವ್ಯಾಖ್ಯಾನಿಸು MKL_FA_WG_ಗಾತ್ರ 256

ಬಳಸಿ oneapi::mkl::transpose;
ಬಳಸಿ oneapi::mkl::blas::ಸ್ತಂಭ_major::gemm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack all GQA Q heads for one KV head into fp16, applying q_scale.
// Launches one kernel per GQA group — each kernel copies exactly
// n_queries * DKQ elements using the per-group dst offset and
// per-head source stride.
ಸ್ಥಿರ ಶೂನ್ಯ mkl_fa_pack_q_fp16(
    dpct::ಸರತಿ_ptr stream,
    sycl::half * __restrict dst,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict q_src,
    ಪೂರ್ಣಾಂಕ n_queries, ಪೂರ್ಣಾಂಕ DKQ,
    ಪೂರ್ಣಾಂಕ gqa_ಅನುಪಾತ, ಪೂರ್ಣಾಂಕ kvh_ತಳ_ತಲೆ,
    ದಶಮಾಂಶ q_scale, int64_t q_ಅಡ್ಡಸಾಲು_stride, int64_t q_ತಲೆ_stride,
    int64_t wg_ಗಾತ್ರ) {

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ iqg = 0; iqg < gqa_ಅನುಪಾತ; iqg++) {
        ಪೂರ್ಣಾಂಕ     iqh       = kvh_ತಳ_ತಲೆ + iqg;
        sycl::half * dst_g = dst + (int64_t)iqg * n_queries * DKQ;

        ಸ್ಥಿರಾಂಕ int64_t n_elem = (int64_t)n_queries * DKQ;
        ಸ್ಥಿರಾಂಕ int64_t wg = ((n_elem + wg_ಗಾತ್ರ - 1) / wg_ಗಾತ್ರ) * wg_ಗಾತ್ರ;

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg, wg_ಗಾತ್ರ),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
                    int64_t e = ಅಂಶ.get_global_id(0);
                    ಆದರೆ (e >= n_elem) ಹಿಂತಿರುಗಿ;

                    int64_t q = e / DKQ;
                    int64_t d = e - q * DKQ;

                    // Stride-aware source offset: handles permuted,
                    // sliced, or contiguous Q tensor layouts.
                    int64_t src_off = d
                        + q * q_ಅಡ್ಡಸಾಲು_stride
                        + (int64_t)iqh * q_ತಲೆ_stride;

                    dst_g[e] = sycl::half(
                        q_src[src_off] * q_scale);
                });
        });
    }
}

// Zero-initialize the online softmax state arrays.
// KQ_max → -inf, KQ_sum → 0, VKQ_accum → 0.
// Merged into one kernel to avoid per-array launch overhead.
ಸ್ಥಿರ ಶೂನ್ಯ mkl_fa_init_softmax_state(
    dpct::ಸರತಿ_ptr stream,
    ದಶಮಾಂಶ * kmax, ದಶಮಾಂಶ * ksum, ದಶಮಾಂಶ * vacc,
    ಪೂರ್ಣಾಂಕ n_query_ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ DV, int64_t wg_ಗಾತ್ರ) {

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    neg_inf   = -1e30f;
    ಸ್ಥಿರಾಂಕ int64_t  n_maxsum  = n_query_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t  n_vacc    = (int64_t)n_query_ಅಡ್ಡಸಾಲುಗಳು * DV;
    ಸ್ಥಿರಾಂಕ int64_t  ಒಟ್ಟು     = (n_vacc > n_maxsum) ? n_vacc : n_maxsum;
    ಸ್ಥಿರಾಂಕ int64_t  wg = ((ಒಟ್ಟು + wg_ಗಾತ್ರ - 1) / wg_ಗಾತ್ರ) * wg_ಗಾತ್ರ;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_ಗಾತ್ರ),
            [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
                int64_t i = ಅಂಶ.get_global_id(0);
                ಆದರೆ (i < n_maxsum) {
                    kmax[i] = neg_inf;
                    ksum[i] = 0.0f;
                }
                ಆದರೆ (i < n_vacc) {
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
ಸ್ಥಿರ ಶೂನ್ಯ mkl_fa_online_softmax_chunk(
    dpct::ಸರತಿ_ptr stream,
    ದಶಮಾಂಶ * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    ದಶಮಾಂಶ * __restrict KQ_max,
    ದಶಮಾಂಶ * __restrict KQ_ಮೊತ್ತ,
    ದಶಮಾಂಶ * __restrict VKQ_accum,
    ಪೂರ್ಣಾಂಕ q0, ಪೂರ್ಣಾಂಕ q_ಅಡ್ಡಸಾಲುಗಳು, ಪೂರ್ಣಾಂಕ n_queries, ಪೂರ್ಣಾಂಕ DV,
    ಪೂರ್ಣಾಂಕ chunk_ಗಾತ್ರ, ಪೂರ್ಣಾಂಕ chunk_ಆರಂಭ,
    ಪೂರ್ಣಾಂಕ kvh_ತಲೆ,
    ಸ್ಥಿರಾಂಕ sycl::half * ಮುಖವಾಡ_ದತ್ತಾಂಶ, int64_t ಮುಖವಾಡ_ತಲೆ_stride,
    int64_t ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_stride, ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ_n_heads,
    ದಶಮಾಂಶ logit_softcap, int64_t wg_ಗಾತ್ರ) {

    // One work-group per query row: exactly q_rows groups of wg_size
    // items. q_rows * wg_size is already a multiple of wg_size, so unlike
    // the one-item-per-row kernels there is no round-up / tail guard.
    ಸ್ಥಿರಾಂಕ int64_t wg         = q_ಅಡ್ಡಸಾಲುಗಳು * wg_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     local_ಗಾತ್ರ = (ಪೂರ್ಣಾಂಕ) wg_ಗಾತ್ರ;  // stride in the loops below
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_ಗಾತ್ರ),
            [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ local_id = (ಪೂರ್ಣಾಂಕ)ಅಂಶ.get_local_id(0);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = (ಪೂರ್ಣಾಂಕ)ಅಂಶ.get_ಗುಂಪು(0); // tile-relative
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jc_abs    = q0 + ಅಡ್ಡಸಾಲು;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ gqa_ಗುಂಪು = jc_abs / n_queries;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q_ಅಡ್ಡಸಾಲು     = jc_abs % n_queries;
                // Score buffers are tile-local (relative index).
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict KQ_ಅಡ್ಡಸಾಲು = KQ_f32
                    + ಅಡ್ಡಸಾಲು * (int64_t)chunk_ಗಾತ್ರ;
                sycl::half * __restrict S_ಅಡ್ಡಸಾಲು = S_f16
                    + ಅಡ್ಡಸಾಲು * (int64_t)chunk_ಗಾತ್ರ;
                // Persistent accumulator is full-sized (absolute index).
                ದಶಮಾಂಶ * __restrict vkq = VKQ_accum
                    + jc_abs * (int64_t)DV;
                ಸ್ಥಿರಾಂಕ sycl::half * ಮುಖವಾಡ_h = ಶೂನ್ಯಸೂಚಕ;
                int64_t m_stride = 0;
                ಆದರೆ (ಮುಖವಾಡ_ದತ್ತಾಂಶ) {
                    ಪೂರ್ಣಾಂಕ m_ತಲೆ = (ಮುಖವಾಡ_n_heads > 1)
                        ? (kvh_ತಲೆ + gqa_ಗುಂಪು) : 0;
                    ಮುಖವಾಡ_h   = ಮುಖವಾಡ_ದತ್ತಾಂಶ + (int64_t)m_ತಲೆ * ಮುಖವಾಡ_ತಲೆ_stride;
                    m_stride = ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_stride;
                }
                // Score at chunk offset i — original per-element math.
                ಸ್ವಯಂಚಾಲಿತ score = [&](ಪೂರ್ಣಾಂಕ i) {
                    ದಶಮಾಂಶ s = KQ_ಅಡ್ಡಸಾಲು[i];
                    ಆದರೆ (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    ಆದರೆ (ಮುಖವಾಡ_h) {
                        s += (ದಶಮಾಂಶ)ಮುಖವಾಡ_h[q_ಅಡ್ಡಸಾಲು * m_stride
                            + (chunk_ಆರಂಭ + i)];
                    }
                    ಹಿಂತಿರುಗಿ s;
                };
                // Pass 1: strided (coalesced) row-wise local maximum.
                ದಶಮಾಂಶ local_max = -1e30f;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = local_id; i < chunk_ಗಾತ್ರ; i += local_ಗಾತ್ರ) {
                    ದಶಮಾಂಶ s = score(i);
                    ಆದರೆ (s > local_max) local_max = s;
                }
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ final_local_max = sycl::reduce_over_ಗುಂಪು(
                    ಅಂಶ.get_ಗುಂಪು(), local_max, sycl::ಗರಿಷ್ಠ<ದಶಮಾಂಶ>());
                // Rescale previous accumulator by exp(old_max - new_max)
                ದಶಮಾಂಶ old_max = KQ_max[jc_abs];
                ದಶಮಾಂಶ new_max = (old_max > final_local_max) ? old_max : final_local_max;
                ದಶಮಾಂಶ rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::ಘಾತಾಂಕ(old_max - new_max);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = local_id; v < DV; v += local_ಗಾತ್ರ) {
                    vkq[v] *= rescale;
                }
                // Pass 2: softmax numerators, strided; S row written once.
                ದಶಮಾಂಶ local_ಮೊತ್ತ = 0.0f;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = local_id; i < chunk_ಗಾತ್ರ; i += local_ಗಾತ್ರ) {
                    ದಶಮಾಂಶ s = score(i);
                    ದಶಮಾಂಶ val = sycl::native::ಘಾತಾಂಕ(s - new_max);
                    S_ಅಡ್ಡಸಾಲು[i] = sycl::half(val);
                    local_ಮೊತ್ತ += val;
                }
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಒಟ್ಟು_ಮೊತ್ತ = sycl::reduce_over_ಗುಂಪು(
                    ಅಂಶ.get_ಗುಂಪು(), local_ಮೊತ್ತ, sycl::plus<ದಶಮಾಂಶ>());
                ಆದರೆ (local_id == 0) {
                    KQ_ಮೊತ್ತ[jc_abs] = KQ_ಮೊತ್ತ[jc_abs] * rescale + ಒಟ್ಟು_ಮೊತ್ತ;
                    KQ_max[jc_abs] = new_max;
                }
            });
    });
}

// Write one GQA group's normalized output to its destination head.
ಸ್ಥಿರ ಶೂನ್ಯ mkl_fa_normalize_ತಲೆ(
    dpct::ಸರತಿ_ptr stream,
    ದಶಮಾಂಶ * __restrict dst_batch,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict VKQ_accum,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict KQ_ಮೊತ್ತ,
    ಪೂರ್ಣಾಂಕ iqh, ಪೂರ್ಣಾಂಕ n_queries, ಪೂರ್ಣಾಂಕ DV, ಪೂರ್ಣಾಂಕ n_q_heads,
    int64_t src_ಸ್ಥಾನಾಂತರ, int64_t wg_ಗಾತ್ರ) {

    ಸ್ಥಿರಾಂಕ int64_t wg = ((n_queries + wg_ಗಾತ್ರ - 1) / wg_ಗಾತ್ರ) * wg_ಗಾತ್ರ;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_ಗಾತ್ರ),
            [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
                ಪೂರ್ಣಾಂಕ jc = ಅಂಶ.get_global_id(0);
                ಆದರೆ (jc >= n_queries) ಹಿಂತಿರುಗಿ;

                ಪೂರ್ಣಾಂಕ     ksum_idx = (ಪೂರ್ಣಾಂಕ)(src_ಸ್ಥಾನಾಂತರ / DV) + jc;
                ದಶಮಾಂಶ   inv_ಮೊತ್ತ  = 1.0f / KQ_ಮೊತ್ತ[ksum_idx];
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict src = VKQ_accum
                    + src_ಸ್ಥಾನಾಂತರ + jc * (int64_t)DV;
                // Interleaved dst layout (matching TILE):
                // rows alternate between heads, then increment query.
                // offset = (query * n_q_heads + head) * DV
                ದಶಮಾಂಶ * __restrict dst_ಅಡ್ಡಸಾಲು = dst_batch
                    + ((int64_t)jc * n_q_heads + iqh) * (int64_t)DV;

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < DV; v++) {
                    dst_ಅಡ್ಡಸಾಲು[v] = src[v] * inv_ಮೊತ್ತ;
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
ಎಣಿಕೆ mkl_fa_kv_desc_ಬಹುಲಕ {
    MKL_FA_KV_ಬಹುಲಕ_F16_DENSE       = 0,
    MKL_FA_KV_ಬಹುಲಕ_F16_INTERLEAVED = 1,
    MKL_FA_KV_ಬಹುಲಕ_QUANT_CONTIG    = 2,
    MKL_FA_KV_ಬಹುಲಕ_QUANT_NC        = 3,
};

ರಚನೆ mkl_fa_kv_desc {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *         ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
    ggml_type            type = GGML_TYPE_F16;
    int64_t              D    = 0;      // ne[0]
    int64_t              nb1  = 0;      // byte stride, seq dim
    int64_t              nb2  = 0;      // byte stride, head dim
    mkl_fa_kv_desc_ಬಹುಲಕ  mode = MKL_FA_KV_ಬಹುಲಕ_F16_DENSE;
    int64_t              ts   = 0;      // type size (mode 3 base offset)
    int64_t              s01  = 0;      // nc row stride in blocks (mode 3)
    int64_t              s02  = 0;      // nc head stride in blocks (mode 3)
};

ಸ್ಥಿರ mkl_fa_kv_desc mkl_fa_make_desc(ಸ್ಥಿರಾಂಕ ggml_tensor * T, ತಾರ್ಕಿಕ interleaved, ಪೂರ್ಣಾಂಕ n_kv_heads) {
    mkl_fa_kv_desc d;
    d.ದತ್ತಾಂಶ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)T->ದತ್ತಾಂಶ;
    d.type = T->type;
    d.D    = T->ne[0];
    d.nb1  = (int64_t)T->nb[1];
    d.nb2  = (int64_t)T->nb[2];
    d.ts   = (int64_t)ggml_type_ಗಾತ್ರ(T->type);

    ಆದರೆ (T->type == GGML_TYPE_F16) {
        // MLA's V cache is a 512-wide view of 576-wide K rows. Treat any
        // padded row stride as strided even when there is only one KV head.
        d.mode = interleaved || d.nb1 != d.D * (int64_t)ಗಾತ್ರ(sycl::half)
            ? MKL_FA_KV_ಬಹುಲಕ_F16_INTERLEAVED : MKL_FA_KV_ಬಹುಲಕ_F16_DENSE;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_is_contiguously_allocated(T) && !interleaved) {
        d.mode = MKL_FA_KV_ಬಹುಲಕ_QUANT_CONTIG;
    } ಇಲ್ಲದಿದ್ದರೆ {
        d.mode = MKL_FA_KV_ಬಹುಲಕ_QUANT_NC;
        ಸ್ಥಿರಾಂಕ int64_t bs          = (int64_t)ggml_blck_ಗಾತ್ರ(T->type);
        ಸ್ಥಿರಾಂಕ int64_t blk_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = T->ne[0] / bs;
        // True Gemma interleave packs heads within a row (nb[2] < ne[1]*nb[1])
        // → reconstruct physical strides. Padded seq-views (nb[2] > ne[1]*nb[1])
        // already have correct physical strides.
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ gemma = interleaved &&
            ((int64_t)T->nb[2] < (int64_t)T->ne[1] * (int64_t)T->nb[1]);
        ಆದರೆ (gemma) {
            d.s01 = (int64_t)n_kv_heads * blk_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
            d.s02 = blk_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        } ಇಲ್ಲದಿದ್ದರೆ {
            d.s01 = d.nb1 / d.ts;
            d.s02 = d.nb2 / d.ts;
        }
    }
    ಹಿಂತಿರುಗಿ d;
}

// Dequant one KV-head chunk into a dense [this_chunk x D] fp16 buffer.
ಸ್ಥಿರ ಶೂನ್ಯ mkl_fa_dequant_chunk(
    dpct::ಸರತಿ_ptr stream, ಸ್ಥಿರಾಂಕ mkl_fa_kv_desc & d, ggml_tensor * dst_ctx,
    sycl::half * ಹೊರಹೋಗು, ಪೂರ್ಣಾಂಕ ikvh, ಪೂರ್ಣಾಂಕ chunk_ಆರಂಭ, ಪೂರ್ಣಾಂಕ this_chunk) {

    ಸ್ಥಿರಾಂಕ int64_t D = d.D;
    ಆಯ್ಕೆ (d.mode) {
        ಸ್ಥಿತಿ MKL_FA_KV_ಬಹುಲಕ_F16_DENSE: {
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ = d.ದತ್ತಾಂಶ + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_ಆರಂಭ * d.nb1;
            stream->ಸ್ಮರಣೆ_ನಕಲು(ಹೊರಹೋಗು, ತಳ, (size_t)this_chunk * D * ಗಾತ್ರ(sycl::half));
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ MKL_FA_KV_ಬಹುಲಕ_F16_INTERLEAVED: {
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ = d.ದತ್ತಾಂಶ + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_ಆರಂಭ * d.nb1;
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_halfs = d.nb1 / (int64_t)ಗಾತ್ರ(sycl::half);
            ಸ್ಥಿರಾಂಕ sycl::half * src = (ಸ್ಥಿರಾಂಕ sycl::half *)ತಳ;
            stream->parallel_for(
                sycl::range<2>((size_t)this_chunk, (size_t)D),
                [=](sycl::ಅಂಶ<2> it) {
                    int64_t r = it.get_id(0);
                    int64_t c = it.get_id(1);
                    ಹೊರಹೋಗು[r * D + c] = src[r * ಅಡ್ಡಸಾಲು_halfs + c];
                });
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ MKL_FA_KV_ಬಹುಲಕ_QUANT_CONTIG: {
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ = d.ದತ್ತಾಂಶ + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_ಆರಂಭ * d.nb1;
            ಗಮ್ಯ_fp16_sycl_t ಗಮ್ಯ_fp16 = ggml_get_ಗಮ್ಯ_fp16_sycl(d.type, dst_ctx);
            ಗಮ್ಯ_fp16(ತಳ, ಹೊರಹೋಗು, (int64_t)this_chunk * D, stream);
            ನಿಲ್ಲಿಸು;
        }
        ಪೂರ್ವನಿಯೋಜಿತ: {  // MKL_FA_KV_MODE_QUANT_NC
            ಗಮ್ಯ_fp16_nc_sycl_t ಗಮ್ಯ_fp16 = ggml_get_ಗಮ್ಯ_fp16_nc_sycl(d.type);
            ಸ್ಥಿರಾಂಕ int64_t ತಳ_ಖಂಡಗಳು = (int64_t)ikvh * d.s02
                + (int64_t)chunk_ಆರಂಭ * d.s01;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ = d.ದತ್ತಾಂಶ + ತಳ_ಖಂಡಗಳು * d.ts;
            // ne02 = ne03 = 1 → s02/s03 inert; head+chunk offset carried by base.
            ಗಮ್ಯ_fp16(ತಳ, ಹೊರಹೋಗು, D, this_chunk, 1, 1, d.s01, d.s02, d.s02, stream);
            ನಿಲ್ಲಿಸು;
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
ಶೂನ್ಯ ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ಸ್ಥಿರಾಂಕ ggml_tensor * Q    = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * K    = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * V    = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಮುಖವಾಡ = dst->src[3];
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    ದಶಮಾಂಶ scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    ಸ್ಮರಣೆ_ನಕಲು(&scale,         (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)KQV->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&max_bias,      (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)KQV->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)KQV->op_params + 2, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ q_scale = scale;

    // --- Dimensions ---
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DKQ        = (ಪೂರ್ಣಾಂಕ)K->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ DV         = (ಪೂರ್ಣಾಂಕ)V->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_queries  = (ಪೂರ್ಣಾಂಕ)Q->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_q_heads  = (ಪೂರ್ಣಾಂಕ)Q->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_kv_heads = (ಪೂರ್ಣಾಂಕ)K->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_batch    = (ಪೂರ್ಣಾಂಕ)Q->ne[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_kv       = (ಪೂರ್ಣಾಂಕ)K->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ gqa_ಅನುಪಾತ  = n_q_heads / n_kv_heads;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_query_ಅಡ್ಡಸಾಲುಗಳು = n_queries * gqa_ಅನುಪಾತ;

    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    GGML_ASSERT(max_bias == 0.0f);  // ALiBi not supported
    GGML_ASSERT(Q->ne[3] == K->ne[3] || K->ne[3] == 1);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ chunk_ಗಾತ್ರ = std::min(MKL_FA_CHUNK_ಗಾತ್ರ_KV, n_kv);

    // Query rows are processed in tiles of q_tile_rows so the score buffers
    // (KQ_f32/S_f16 = q_tile_rows * chunk_size) stay bounded regardless of
    // batch size. n_query_rows <= Q_TILE is a single tile (no extra work).
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ q_tile_env = ggml_sycl_get_env("GGML_SYCL_MKL_FA_Q_TILE", MKL_FA_Q_TILE);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q_tile_ಅಡ್ಡಸಾಲುಗಳು = std::max(1, std::min(q_tile_env, n_query_ಅಡ್ಡಸಾಲುಗಳು));

    ಸ್ಥಿರಾಂಕ int64_t wg_ಗಾತ್ರ = MKL_FA_WG_ಗಾತ್ರ;

    // --- Debug output (gated by GGML_SYCL_MKL_FA_DEBUG=1) ---
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ mkl_call_ಎಣಿಕೆ = 0;
    mkl_call_ಎಣಿಕೆ++;
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ mkl_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ do_print = (mkl_debug == 1);

    ಸ್ಥಿರಾಂಕ int64_t q_ಅಡ್ಡಸಾಲು_stride  = Q->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ int64_t q_ತಲೆ_stride = Q->nb[2] / ಗಾತ್ರ(ದಶಮಾಂಶ);

    // Alias the dequantized buffers only when K and V expose the same values.
    // MLA V is a narrower view of K and needs its own strided dequantization.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ V_is_K_view = V->ne[0] == K->ne[0] && V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    // Early interleaved detection for debug output.
    // True interleaved detection happens after dequant (nb12_fp16 == nb11_fp16),
    // but we can pre-detect on the original tensor strides.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ k_early_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ v_early_interleaved =
        !V_is_K_view && ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]);

    ಆದರೆ (do_print) {
        GGML_LOG_INFO("[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d buf=%.1fMB%s%s\n",
                mkl_call_ಎಣಿಕೆ, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_ಅನುಪಾತ, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_ಗಾತ್ರ,
                (ದ್ವಿಗುಣ)((int64_t)n_query_ಅಡ್ಡಸಾಲುಗಳು * chunk_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ))
                    / (1024.0 * 1024.0),
                k_early_interleaved ? " K_ILV" : "",
                v_early_interleaved ? " V_ILV" : "");
        GGML_LOG_INFO("[MKL-FA] #%d Q-nb1=%lld Q-nb2=%lld "
                "q_rs=%lld q_hs=%lld dst_rs=%lld dst_hs=%lld\n",
                mkl_call_ಎಣಿಕೆ,
                (ದೀರ್ಘ ದೀರ್ಘ)Q->nb[1], (ದೀರ್ಘ ದೀರ್ಘ)Q->nb[2],
                (ದೀರ್ಘ ದೀರ್ಘ)q_ಅಡ್ಡಸಾಲು_stride, (ದೀರ್ಘ ದೀರ್ಘ)q_ತಲೆ_stride,
                (ದೀರ್ಘ ದೀರ್ಘ)(KQV->nb[1] / ಗಾತ್ರ(ದಶಮಾಂಶ)),
                (ದೀರ್ಘ ದೀರ್ಘ)(KQV->nb[2] / ಗಾತ್ರ(ದಶಮಾಂಶ)));
    }

    // --- Stream and allocators ---
    dpct::ಸರತಿ_ptr stream = ctx.stream();

#ವ್ಯಾಖ್ಯಾನಿಸು MKL_TAKE_ಸಮಯ(t0)  ಸ್ವಯಂಚಾಲಿತ t0 = std::chrono::steady_clock::now()
#ವ್ಯಾಖ್ಯಾನಿಸು MKL_ACCUM(acc, t0) ಮಾಡು { ಆದರೆ (do_print) { \
    acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count(); \
} } ಆಗ(0)

    int64_t gemm_kq_ಸಮಯ_us  = 0;
    int64_t gemm_vkq_ಸಮಯ_us = 0;
    int64_t softmax_ಸಮಯ_us  = 0;
    int64_t dequant_ಸಮಯ_us  = 0;

    MKL_TAKE_ಸಮಯ(t_deq);

    // --- K/V dequant descriptors ---
    // Dequant is done per-chunk inside the KV loop (footprint independent of
    // context). Output is always dense row-major fp16 [this_chunk x D], lda=D.
    // Interleaved detection: ne[1]*nb[1] != nb[2] means heads are interleaved.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ k_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ v_interleaved =
        ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

    ಸ್ಥಿರಾಂಕ mkl_fa_kv_desc K_desc = mkl_fa_make_desc(K, k_interleaved, n_kv_heads);
    ಸ್ಥಿರಾಂಕ mkl_fa_kv_desc V_desc = V_is_K_view
        ? K_desc : mkl_fa_make_desc(V, v_interleaved, n_kv_heads);

    MKL_ACCUM(dequant_ಸಮಯ_us, t_deq);

    // --- Resolve mask pointers ---
    int64_t ಮುಖವಾಡ_ತಲೆ_stride = 0;
    int64_t ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_stride  = 0;
    ಪೂರ್ಣಾಂಕ     ಮುಖವಾಡ_n_heads     = 0;

    ಆದರೆ (ಮುಖವಾಡ) {
        // Use actual fp16 device size (2 bytes), NOT sizeof(sycl::half)
        // which may be 4 on the host in oneAPI.
        ಮುಖವಾಡ_ತಲೆ_stride = ಮುಖವಾಡ->nb[2] / 2;
        ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_stride  = ಮುಖವಾಡ->nb[1] / 2;
        ಮುಖವಾಡ_n_heads     = (ಪೂರ್ಣಾಂಕ)ಮುಖವಾಡ->ne[2];
    }

    // --- Allocate intermediates from pool ---
    ggml_sycl_pool & pool = ctx.pool();

    ggml_sycl_pool_alloc<ದಶಮಾಂಶ>      KQ_f32(pool);      // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);       // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<ದಶಮಾಂಶ>      VKQ_chunk(pool);   // [q_tile_rows x DV]
    ggml_sycl_pool_alloc<ದಶಮಾಂಶ>      VKQ_accum(pool);   // [n_query_rows x DV] (full)
    ggml_sycl_pool_alloc<ದಶಮಾಂಶ>      KQ_max(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<ದಶಮಾಂಶ>      KQ_ಮೊತ್ತ(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<sycl::half> Q_ತಲೆ_f16(pool);  // [n_query_rows x DKQ] (full)
    ggml_sycl_pool_alloc<sycl::half> K_chunk_f16(pool); // [chunk x DKQ] (per-chunk dequant)
    ggml_sycl_pool_alloc<sycl::half> V_chunk_f16(pool); // [chunk x DV] (per-chunk dequant)

    KQ_f32.alloc((size_t)q_tile_ಅಡ್ಡಸಾಲುಗಳು * chunk_ಗಾತ್ರ);
    S_f16.alloc((size_t)q_tile_ಅಡ್ಡಸಾಲುಗಳು * chunk_ಗಾತ್ರ);
    VKQ_chunk.alloc((size_t)q_tile_ಅಡ್ಡಸಾಲುಗಳು * DV);
    VKQ_accum.alloc((size_t)n_query_ಅಡ್ಡಸಾಲುಗಳು * DV);
    KQ_max.alloc(n_query_ಅಡ್ಡಸಾಲುಗಳು);
    KQ_ಮೊತ್ತ.alloc(n_query_ಅಡ್ಡಸಾಲುಗಳು);
    Q_ತಲೆ_f16.alloc((size_t)n_query_ಅಡ್ಡಸಾಲುಗಳು * DKQ);
    K_chunk_f16.alloc((size_t)chunk_ಗಾತ್ರ * DKQ);

    sycl::half * V_chunk_f16_ptr;
    ಆದರೆ (V_is_K_view) {
        V_chunk_f16_ptr = K_chunk_f16.ptr;   // V aliases K (DV == DKQ)
    } ಇಲ್ಲದಿದ್ದರೆ {
        V_chunk_f16.alloc((size_t)chunk_ಗಾತ್ರ * DV);
        V_chunk_f16_ptr = V_chunk_f16.ptr;
    }

    sycl::half * Q_ತಲೆ_f16_ptr  = Q_ತಲೆ_f16.ptr;
    ದಶಮಾಂಶ      * KQ_f32_ptr      = KQ_f32.ptr;
    sycl::half * S_f16_ptr       = S_f16.ptr;
    ದಶಮಾಂಶ      * VKQ_chunk_ptr   = VKQ_chunk.ptr;
    ದಶಮಾಂಶ      * VKQ_accum_ptr   = VKQ_accum.ptr;
    ದಶಮಾಂಶ      * KQ_max_ptr      = KQ_max.ptr;
    ದಶಮಾಂಶ      * KQ_ಮೊತ್ತ_ptr      = KQ_ಮೊತ್ತ.ptr;
    sycl::half * K_chunk_f16_ptr = K_chunk_f16.ptr;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = 1.0f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta  = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = 0; ib < n_batch; ib++) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * Q_batch = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)Q->ದತ್ತಾಂಶ
            + ib * (Q->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ));
        ದಶಮಾಂಶ * dst_batch = (ದಶಮಾಂಶ *)KQV->ದತ್ತಾಂಶ
            + ib * (KQV->nb[3] / ಗಾತ್ರ(ದಶಮಾಂಶ));

        ಸ್ಥಿರಾಂಕ sycl::half * ಮುಖವಾಡ_batch = ಶೂನ್ಯಸೂಚಕ;
        ಆದರೆ (ಮುಖವಾಡ) {
            ಪೂರ್ಣಾಂಕ m_batch = (ಮುಖವಾಡ->ne[3] > 1) ? ib : 0;
            ಮುಖವಾಡ_batch = (ಸ್ಥಿರಾಂಕ sycl::half *)ಮುಖವಾಡ->ದತ್ತಾಂಶ
                + m_batch * (ಮುಖವಾಡ->nb[3] / 2);  // 2 = actual fp16 device size
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            ಪೂರ್ಣಾಂಕ kvh_ತಳ_ತಲೆ = ikvh * gqa_ಅನುಪಾತ;

            // 1. Pack all GQA Q heads into fp16 (full n_query_rows)
            mkl_fa_pack_q_fp16(stream,
                Q_ತಲೆ_f16_ptr, Q_batch,
                n_queries, DKQ,
                gqa_ಅನುಪಾತ, kvh_ತಳ_ತಲೆ,
                q_scale, q_ಅಡ್ಡಸಾಲು_stride, q_ತಲೆ_stride, wg_ಗಾತ್ರ);

            // 2. Initialize softmax state (full n_query_rows)
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_ಮೊತ್ತ_ptr, VKQ_accum_ptr,
                n_query_ಅಡ್ಡಸಾಲುಗಳು, DV, wg_ಗಾತ್ರ);

            // Sync before MKL GEMM (MKL may use an internal queue)
            stream->wait();

            // 3. KV chunk loop (OUTER): dequant each chunk once, then tile queries.
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ chunk_ಆರಂಭ = 0; chunk_ಆರಂಭ < n_kv; chunk_ಆರಂಭ += chunk_ಗಾತ್ರ) {
                ಪೂರ್ಣಾಂಕ this_chunk = std::min(chunk_ಗಾತ್ರ, n_kv - chunk_ಆರಂಭ);

                // 3a. Dequant this KV chunk to dense fp16 (once per chunk)
                {
                    MKL_TAKE_ಸಮಯ(t0);
                    mkl_fa_dequant_chunk(stream, K_desc, KQV,
                        K_chunk_f16_ptr, ikvh, chunk_ಆರಂಭ, this_chunk);
                    ಆದರೆ (!V_is_K_view) {
                        mkl_fa_dequant_chunk(stream, V_desc, KQV,
                            V_chunk_f16_ptr, ikvh, chunk_ಆರಂಭ, this_chunk);
                    }
                    stream->wait();  // dequant must be ready before MKL GEMM
                    MKL_ACCUM(dequant_ಸಮಯ_us, t0);
                }

                // 3b. Query tile loop (INNER) — bounds KQ_f32/S_f16 footprint.
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ q0 = 0; q0 < n_query_ಅಡ್ಡಸಾಲುಗಳು; q0 += q_tile_ಅಡ್ಡಸಾಲುಗಳು) {
                    ಪೂರ್ಣಾಂಕ q_ಅಡ್ಡಸಾಲುಗಳು = std::min(q_tile_ಅಡ್ಡಸಾಲುಗಳು, n_query_ಅಡ್ಡಸಾಲುಗಳು - q0);

                    // GEMM: KQ = Q_tile × K_chunk^T
                    {
                        MKL_TAKE_ಸಮಯ(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::trans, transpose::nontrans,
                            this_chunk, q_ಅಡ್ಡಸಾಲುಗಳು, DKQ,
                            alpha,
                            K_chunk_f16_ptr, DKQ,
                            Q_ತಲೆ_f16_ptr + (int64_t)q0 * DKQ, DKQ,
                            beta,
                            KQ_f32_ptr, this_chunk);
                        ಪ್ರಯತ್ನಿಸು { ev.wait_and_throw(); } ಹಿಡಿ (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM KQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM KQ failed");
                        }
                        MKL_ACCUM(gemm_kq_ಸಮಯ_us, t0);
                    }
                    // Online softmax over this chunk for this query tile
                    {
                        MKL_TAKE_ಸಮಯ(t0);
                        mkl_fa_online_softmax_chunk(stream,
                            KQ_f32_ptr, S_f16_ptr,
                            KQ_max_ptr, KQ_ಮೊತ್ತ_ptr, VKQ_accum_ptr,
                            q0, q_ಅಡ್ಡಸಾಲುಗಳು, n_queries, DV,
                            this_chunk, chunk_ಆರಂಭ,
                            kvh_ತಳ_ತಲೆ,
                            ಮುಖವಾಡ_batch, ಮುಖವಾಡ_ತಲೆ_stride,
                            ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_stride, ಮುಖವಾಡ_n_heads,
                            logit_softcap, wg_ಗಾತ್ರ);
                        stream->wait();  // S_f16 must be ready for GEMM
                        MKL_ACCUM(softmax_ಸಮಯ_us, t0);
                    }

                    // GEMM: VKQ_chunk = S × V_chunk
                    {
                        MKL_TAKE_ಸಮಯ(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::nontrans, transpose::nontrans,
                            DV, q_ಅಡ್ಡಸಾಲುಗಳು, this_chunk,
                            alpha,
                            V_chunk_f16_ptr, DV,
                            S_f16_ptr, this_chunk,
                            beta,
                            VKQ_chunk_ptr, DV);
                        ಪ್ರಯತ್ನಿಸು { ev.wait_and_throw(); } ಹಿಡಿ (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM VKQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM VKQ failed");
                        }
                        MKL_ACCUM(gemm_vkq_ಸಮಯ_us, t0);
                    }
                    // VKQ_accum[q0..] += VKQ_chunk
                    {
                        ಸ್ಥಿರಾಂಕ int64_t n_ಒಟ್ಟು = (int64_t)q_ಅಡ್ಡಸಾಲುಗಳು * DV;
                        ಸ್ಥಿರಾಂಕ int64_t wg = ((n_ಒಟ್ಟು + wg_ಗಾತ್ರ - 1) / wg_ಗಾತ್ರ)
                            * wg_ಗಾತ್ರ;
                        ದಶಮಾಂಶ * accum = VKQ_accum_ptr + (int64_t)q0 * DV;
                        stream->submit([&](sycl::handler & cgh) {
                            cgh.parallel_for(sycl::nd_range<1>(wg, wg_ಗಾತ್ರ),
                                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
                                    int64_t i = ಅಂಶ.get_global_id(0);
                                    ಆದರೆ (i < n_ಒಟ್ಟು) {
                                        accum[i] += VKQ_chunk_ptr[i];
                                    }
                                });
                        });
                    }
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ iqg = 0; iqg < gqa_ಅನುಪಾತ; iqg++) {
                ಪೂರ್ಣಾಂಕ     iqh        = kvh_ತಳ_ತಲೆ + iqg;
                int64_t src_ಸ್ಥಾನಾಂತರ = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_ತಲೆ(stream,
                    dst_batch, VKQ_accum_ptr, KQ_ಮೊತ್ತ_ptr,
                    iqh, n_queries, DV, n_q_heads,
                    src_ಸ್ಥಾನಾಂತರ, wg_ಗಾತ್ರ);
            }
        }
    }

#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು MKL_TAKE_ಸಮಯ
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು MKL_ACCUM

    ಆದರೆ (do_print) {
        ಸ್ಥಿರಾಂಕ int64_t v_chunk_elems = V_is_K_view ? 0 : (int64_t)chunk_ಗಾತ್ರ * DV;
        ದ್ವಿಗುಣ ಒಟ್ಟು_mb = (ದ್ವಿಗುಣ)(
            (int64_t)q_tile_ಅಡ್ಡಸಾಲುಗಳು * chunk_ಗಾತ್ರ * ಗಾತ್ರ(ದಶಮಾಂಶ)      // KQ_f32
          + (int64_t)q_tile_ಅಡ್ಡಸಾಲುಗಳು * chunk_ಗಾತ್ರ * ಗಾತ್ರ(sycl::half) // S_f16
          + (int64_t)q_tile_ಅಡ್ಡಸಾಲುಗಳು * DV * ಗಾತ್ರ(ದಶಮಾಂಶ)              // VKQ_chunk
          + (int64_t)n_query_ಅಡ್ಡಸಾಲುಗಳು * DV * ಗಾತ್ರ(ದಶಮಾಂಶ)             // VKQ_accum
          + (int64_t)n_query_ಅಡ್ಡಸಾಲುಗಳು * ಗಾತ್ರ(ದಶಮಾಂಶ)                  // KQ_max
          + (int64_t)n_query_ಅಡ್ಡಸಾಲುಗಳು * ಗಾತ್ರ(ದಶಮಾಂಶ)                  // KQ_sum
          + (int64_t)n_query_ಅಡ್ಡಸಾಲುಗಳು * DKQ * ಗಾತ್ರ(sycl::half)       // Q_head_f16
          + (int64_t)chunk_ಗಾತ್ರ * DKQ * ಗಾತ್ರ(sycl::half)         // K_chunk_f16
          + v_chunk_elems * (int64_t)ಗಾತ್ರ(sycl::half)            // V_chunk_f16
        ) / (1024.0 * 1024.0);
        GGML_LOG_INFO("[MKL-FA] #%d n_kv=%d n_q=%d q_tile=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld "
                "buf_mb=%.1f\n",
                mkl_call_ಎಣಿಕೆ, n_kv, n_queries, q_tile_ಅಡ್ಡಸಾಲುಗಳು,
                (ದೀರ್ಘ ದೀರ್ಘ)dequant_ಸಮಯ_us,
                (ದೀರ್ಘ ದೀರ್ಘ)gemm_kq_ಸಮಯ_us,
                (ದೀರ್ಘ ದೀರ್ಘ)softmax_ಸಮಯ_us,
                (ದೀರ್ಘ ದೀರ್ಘ)gemm_vkq_ಸಮಯ_us,
                ಒಟ್ಟು_mb);
    }
}
