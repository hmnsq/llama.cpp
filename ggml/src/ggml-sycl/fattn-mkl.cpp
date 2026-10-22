// Flash attention via oneMKL GEMM (XMX-accelerated).
// Uses column_major::gemm for Q*K^T and S*V matmuls
// with an online softmax SYCL kernel.
//
// All GQA query heads sharing a KV head are batched into single
// GEMM calls, amortizing MKL launch overhead across K and V reuse.
//

#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "fattn-common.hpp"
#অন্তর্ভাব "convert.hpp"
#অন্তর্ভাব "fattn.hpp"

#অন্তর্ভাব <oneapi/mkl.hpp>
#অন্তর্ভাব <cstdio>
#অন্তর্ভাব <chrono>

#define MKL_FA_CHUNK_আকার_KV 8192

// Number of query rows processed per tile. The score buffers (KQ_f32, S_f16)
// are sized q_tile_rows * chunk_size, so this bounds their footprint
// regardless of batch size (n_query_rows = n_queries * gqa_ratio). A typical
// single-ubatch prefill (e.g. ubatch 1024 * gqa 8 = 8192 rows) is exactly one
// tile, so it runs with no extra iterations. Larger batches tile and stay
// bounded. Override with GGML_SYCL_MKL_FA_Q_TILE.
#define MKL_FA_Q_TILE 8192

#define MKL_FA_WG_আকার 256

using oneapi::mkl::transpose;
using oneapi::mkl::blas::column_major::gemm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack all GQA Q heads for one KV head into fp16, applying q_scale.
// Launches one kernel per GQA group — each kernel copies exactly
// n_queries * DKQ elements using the per-group dst offset and
// per-head source stride.
স্থির শূন্য mkl_fa_pack_q_fp16(
    dpct::queue_ptr stream,
    sycl::half * __restrict dst,
    ধ্রুবক দশমিক * __restrict q_src,
    পূর্ণসংখ্যা n_queries, পূর্ণসংখ্যা DKQ,
    পূর্ণসংখ্যা gqa_ratio, পূর্ণসংখ্যা kvh_base_শীর্ষ,
    দশমিক q_scale, int64_t q_row_stride, int64_t q_শীর্ষ_stride,
    int64_t wg_আকার) {

    জন্য (পূর্ণসংখ্যা iqg = 0; iqg < gqa_ratio; iqg++) {
        পূর্ণসংখ্যা     iqh       = kvh_base_শীর্ষ + iqg;
        sycl::half * dst_g = dst + (int64_t)iqg * n_queries * DKQ;

        ধ্রুবক int64_t n_elem = (int64_t)n_queries * DKQ;
        ধ্রুবক int64_t wg = ((n_elem + wg_আকার - 1) / wg_আকার) * wg_আকার;

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg, wg_আকার),
                [=](sycl::nd_item<1> item) {
                    int64_t e = item.get_global_id(0);
                    যদি (e >= n_elem) ফেরত;

                    int64_t q = e / DKQ;
                    int64_t d = e - q * DKQ;

                    // Stride-aware source offset: handles permuted,
                    // sliced, or contiguous Q tensor layouts.
                    int64_t src_off = d
                        + q * q_row_stride
                        + (int64_t)iqh * q_শীর্ষ_stride;

                    dst_g[e] = sycl::half(
                        q_src[src_off] * q_scale);
                });
        });
    }
}

// Zero-initialize the online softmax state arrays.
// KQ_max → -inf, KQ_sum → 0, VKQ_accum → 0.
// Merged into one kernel to avoid per-array launch overhead.
স্থির শূন্য mkl_fa_init_softmax_state(
    dpct::queue_ptr stream,
    দশমিক * kmax, দশমিক * ksum, দশমিক * vacc,
    পূর্ণসংখ্যা n_query_rows, পূর্ণসংখ্যা DV, int64_t wg_আকার) {

    ধ্রুবক দশমিক    neg_inf   = -1e30f;
    ধ্রুবক int64_t  n_maxsum  = n_query_rows;
    ধ্রুবক int64_t  n_vacc    = (int64_t)n_query_rows * DV;
    ধ্রুবক int64_t  যোগ     = (n_vacc > n_maxsum) ? n_vacc : n_maxsum;
    ধ্রুবক int64_t  wg = ((যোগ + wg_আকার - 1) / wg_আকার) * wg_আকার;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_আকার),
            [=](sycl::nd_item<1> item) {
                int64_t i = item.get_global_id(0);
                যদি (i < n_maxsum) {
                    kmax[i] = neg_inf;
                    ksum[i] = 0.0f;
                }
                যদি (i < n_vacc) {
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
স্থির শূন্য mkl_fa_online_softmax_chunk(
    dpct::queue_ptr stream,
    দশমিক * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    দশমিক * __restrict KQ_max,
    দশমিক * __restrict KQ_যোগ,
    দশমিক * __restrict VKQ_accum,
    পূর্ণসংখ্যা q0, পূর্ণসংখ্যা q_rows, পূর্ণসংখ্যা n_queries, পূর্ণসংখ্যা DV,
    পূর্ণসংখ্যা chunk_আকার, পূর্ণসংখ্যা chunk_start,
    পূর্ণসংখ্যা kvh_শীর্ষ,
    ধ্রুবক sycl::half * mask_তথ্য, int64_t mask_শীর্ষ_stride,
    int64_t mask_row_stride, পূর্ণসংখ্যা mask_n_heads,
    দশমিক logit_softcap, int64_t wg_আকার) {

    // One work-group per query row: exactly q_rows groups of wg_size
    // items. q_rows * wg_size is already a multiple of wg_size, so unlike
    // the one-item-per-row kernels there is no round-up / tail guard.
    ধ্রুবক int64_t wg         = q_rows * wg_আকার;
    ধ্রুবক পূর্ণসংখ্যা     local_আকার = (পূর্ণসংখ্যা) wg_আকার;  // stride in the loops below
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_আকার),
            [=](sycl::nd_item<1> item) {
                ধ্রুবক পূর্ণসংখ্যা local_id = (পূর্ণসংখ্যা)item.get_local_id(0);
                ধ্রুবক পূর্ণসংখ্যা row = (পূর্ণসংখ্যা)item.get_group(0); // tile-relative
                ধ্রুবক পূর্ণসংখ্যা jc_abs    = q0 + row;
                ধ্রুবক পূর্ণসংখ্যা gqa_group = jc_abs / n_queries;
                ধ্রুবক পূর্ণসংখ্যা q_row     = jc_abs % n_queries;
                // Score buffers are tile-local (relative index).
                ধ্রুবক দশমিক * __restrict KQ_row = KQ_f32
                    + row * (int64_t)chunk_আকার;
                sycl::half * __restrict S_row = S_f16
                    + row * (int64_t)chunk_আকার;
                // Persistent accumulator is full-sized (absolute index).
                দশমিক * __restrict vkq = VKQ_accum
                    + jc_abs * (int64_t)DV;
                ধ্রুবক sycl::half * mask_h = nullptr;
                int64_t m_stride = 0;
                যদি (mask_তথ্য) {
                    পূর্ণসংখ্যা m_শীর্ষ = (mask_n_heads > 1)
                        ? (kvh_শীর্ষ + gqa_group) : 0;
                    mask_h   = mask_তথ্য + (int64_t)m_শীর্ষ * mask_শীর্ষ_stride;
                    m_stride = mask_row_stride;
                }
                // Score at chunk offset i — original per-element math.
                auto score = [&](পূর্ণসংখ্যা i) {
                    দশমিক s = KQ_row[i];
                    যদি (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    যদি (mask_h) {
                        s += (দশমিক)mask_h[q_row * m_stride
                            + (chunk_start + i)];
                    }
                    ফেরত s;
                };
                // Pass 1: strided (coalesced) row-wise local maximum.
                দশমিক local_max = -1e30f;
                জন্য (পূর্ণসংখ্যা i = local_id; i < chunk_আকার; i += local_আকার) {
                    দশমিক s = score(i);
                    যদি (s > local_max) local_max = s;
                }
                ধ্রুবক দশমিক final_local_max = sycl::reduce_over_group(
                    item.get_group(), local_max, sycl::maximum<দশমিক>());
                // Rescale previous accumulator by exp(old_max - new_max)
                দশমিক old_max = KQ_max[jc_abs];
                দশমিক new_max = (old_max > final_local_max) ? old_max : final_local_max;
                দশমিক rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::exp(old_max - new_max);
                জন্য (পূর্ণসংখ্যা v = local_id; v < DV; v += local_আকার) {
                    vkq[v] *= rescale;
                }
                // Pass 2: softmax numerators, strided; S row written once.
                দশমিক local_যোগ = 0.0f;
                জন্য (পূর্ণসংখ্যা i = local_id; i < chunk_আকার; i += local_আকার) {
                    দশমিক s = score(i);
                    দশমিক val = sycl::native::exp(s - new_max);
                    S_row[i] = sycl::half(val);
                    local_যোগ += val;
                }
                ধ্রুবক দশমিক যোগ_যোগ = sycl::reduce_over_group(
                    item.get_group(), local_যোগ, sycl::plus<দশমিক>());
                যদি (local_id == 0) {
                    KQ_যোগ[jc_abs] = KQ_যোগ[jc_abs] * rescale + যোগ_যোগ;
                    KQ_max[jc_abs] = new_max;
                }
            });
    });
}

// Write one GQA group's normalized output to its destination head.
স্থির শূন্য mkl_fa_normalize_শীর্ষ(
    dpct::queue_ptr stream,
    দশমিক * __restrict dst_batch,
    ধ্রুবক দশমিক * __restrict VKQ_accum,
    ধ্রুবক দশমিক * __restrict KQ_যোগ,
    পূর্ণসংখ্যা iqh, পূর্ণসংখ্যা n_queries, পূর্ণসংখ্যা DV, পূর্ণসংখ্যা n_q_heads,
    int64_t src_offset, int64_t wg_আকার) {

    ধ্রুবক int64_t wg = ((n_queries + wg_আকার - 1) / wg_আকার) * wg_আকার;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_আকার),
            [=](sycl::nd_item<1> item) {
                পূর্ণসংখ্যা jc = item.get_global_id(0);
                যদি (jc >= n_queries) ফেরত;

                পূর্ণসংখ্যা     ksum_idx = (পূর্ণসংখ্যা)(src_offset / DV) + jc;
                দশমিক   inv_যোগ  = 1.0f / KQ_যোগ[ksum_idx];
                ধ্রুবক দশমিক * __restrict src = VKQ_accum
                    + src_offset + jc * (int64_t)DV;
                // Interleaved dst layout (matching TILE):
                // rows alternate between heads, then increment query.
                // offset = (query * n_q_heads + head) * DV
                দশমিক * __restrict dst_row = dst_batch
                    + ((int64_t)jc * n_q_heads + iqh) * (int64_t)DV;

                জন্য (পূর্ণসংখ্যা v = 0; v < DV; v++) {
                    dst_row[v] = src[v] * inv_যোগ;
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
গণনা mkl_fa_kv_desc_mode {
    MKL_FA_KV_MODE_F16_DENSE       = 0,
    MKL_FA_KV_MODE_F16_INTERLEAVED = 1,
    MKL_FA_KV_MODE_QUANT_CONTIG    = 2,
    MKL_FA_KV_MODE_QUANT_NC        = 3,
};

কাঠামো mkl_fa_kv_desc {
    ধ্রুবক অক্ষর *         তথ্য = nullptr;
    ggml_type            type = GGML_TYPE_F16;
    int64_t              D    = 0;      // ne[0]
    int64_t              nb1  = 0;      // byte stride, seq dim
    int64_t              nb2  = 0;      // byte stride, head dim
    mkl_fa_kv_desc_mode  mode = MKL_FA_KV_MODE_F16_DENSE;
    int64_t              ts   = 0;      // type size (mode 3 base offset)
    int64_t              s01  = 0;      // nc row stride in blocks (mode 3)
    int64_t              s02  = 0;      // nc head stride in blocks (mode 3)
};

স্থির mkl_fa_kv_desc mkl_fa_make_desc(ধ্রুবক ggml_tensor * T, যৌক্তিক interleaved, পূর্ণসংখ্যা n_kv_heads) {
    mkl_fa_kv_desc d;
    d.তথ্য = (ধ্রুবক অক্ষর *)T->তথ্য;
    d.type = T->type;
    d.D    = T->ne[0];
    d.nb1  = (int64_t)T->nb[1];
    d.nb2  = (int64_t)T->nb[2];
    d.ts   = (int64_t)ggml_type_আকার(T->type);

    যদি (T->type == GGML_TYPE_F16) {
        // MLA's V cache is a 512-wide view of 576-wide K rows. Treat any
        // padded row stride as strided even when there is only one KV head.
        d.mode = interleaved || d.nb1 != d.D * (int64_t)আকার(sycl::half)
            ? MKL_FA_KV_MODE_F16_INTERLEAVED : MKL_FA_KV_MODE_F16_DENSE;
    } অন্যথা যদি (ggml_is_contiguously_allocated(T) && !interleaved) {
        d.mode = MKL_FA_KV_MODE_QUANT_CONTIG;
    } অন্যথা {
        d.mode = MKL_FA_KV_MODE_QUANT_NC;
        ধ্রুবক int64_t bs          = (int64_t)ggml_blck_আকার(T->type);
        ধ্রুবক int64_t blk_per_row = T->ne[0] / bs;
        // True Gemma interleave packs heads within a row (nb[2] < ne[1]*nb[1])
        // → reconstruct physical strides. Padded seq-views (nb[2] > ne[1]*nb[1])
        // already have correct physical strides.
        ধ্রুবক যৌক্তিক gemma = interleaved &&
            ((int64_t)T->nb[2] < (int64_t)T->ne[1] * (int64_t)T->nb[1]);
        যদি (gemma) {
            d.s01 = (int64_t)n_kv_heads * blk_per_row;
            d.s02 = blk_per_row;
        } অন্যথা {
            d.s01 = d.nb1 / d.ts;
            d.s02 = d.nb2 / d.ts;
        }
    }
    ফেরত d;
}

// Dequant one KV-head chunk into a dense [this_chunk x D] fp16 buffer.
স্থির শূন্য mkl_fa_dequant_chunk(
    dpct::queue_ptr stream, ধ্রুবক mkl_fa_kv_desc & d, ggml_tensor * dst_ctx,
    sycl::half * out, পূর্ণসংখ্যা ikvh, পূর্ণসংখ্যা chunk_start, পূর্ণসংখ্যা this_chunk) {

    ধ্রুবক int64_t D = d.D;
    নির্বাচন (d.mode) {
        ক্ষেত্র MKL_FA_KV_MODE_F16_DENSE: {
            ধ্রুবক অক্ষর * base = d.তথ্য + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            stream->memcpy(out, base, (size_t)this_chunk * D * আকার(sycl::half));
            বিরতি;
        }
        ক্ষেত্র MKL_FA_KV_MODE_F16_INTERLEAVED: {
            ধ্রুবক অক্ষর * base = d.তথ্য + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            ধ্রুবক int64_t row_halfs = d.nb1 / (int64_t)আকার(sycl::half);
            ধ্রুবক sycl::half * src = (ধ্রুবক sycl::half *)base;
            stream->parallel_for(
                sycl::range<2>((size_t)this_chunk, (size_t)D),
                [=](sycl::item<2> it) {
                    int64_t r = it.get_id(0);
                    int64_t c = it.get_id(1);
                    out[r * D + c] = src[r * row_halfs + c];
                });
            বিরতি;
        }
        ক্ষেত্র MKL_FA_KV_MODE_QUANT_CONTIG: {
            ধ্রুবক অক্ষর * base = d.তথ্য + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(d.type, dst_ctx);
            to_fp16(base, out, (int64_t)this_chunk * D, stream);
            বিরতি;
        }
        পূর্বনির্ধারিত: {  // MKL_FA_KV_MODE_QUANT_NC
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(d.type);
            ধ্রুবক int64_t base_blocks = (int64_t)ikvh * d.s02
                + (int64_t)chunk_start * d.s01;
            ধ্রুবক অক্ষর * base = d.তথ্য + base_blocks * d.ts;
            // ne02 = ne03 = 1 → s02/s03 inert; head+chunk offset carried by base.
            to_fp16(base, out, D, this_chunk, 1, 1, d.s01, d.s02, d.s02, stream);
            বিরতি;
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
শূন্য ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ধ্রুবক ggml_tensor * Q    = dst->src[0];
    ধ্রুবক ggml_tensor * K    = dst->src[1];
    ধ্রুবক ggml_tensor * V    = dst->src[2];
    ধ্রুবক ggml_tensor * mask = dst->src[3];
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    দশমিক scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (ধ্রুবক দশমিক *)KQV->op_params + 0, আকার(দশমিক));
    memcpy(&max_bias,      (ধ্রুবক দশমিক *)KQV->op_params + 1, আকার(দশমিক));
    memcpy(&logit_softcap, (ধ্রুবক দশমিক *)KQV->op_params + 2, আকার(দশমিক));

    ধ্রুবক দশমিক q_scale = scale;

    // --- Dimensions ---
    ধ্রুবক পূর্ণসংখ্যা DKQ        = (পূর্ণসংখ্যা)K->ne[0];
    ধ্রুবক পূর্ণসংখ্যা DV         = (পূর্ণসংখ্যা)V->ne[0];
    ধ্রুবক পূর্ণসংখ্যা n_queries  = (পূর্ণসংখ্যা)Q->ne[1];
    ধ্রুবক পূর্ণসংখ্যা n_q_heads  = (পূর্ণসংখ্যা)Q->ne[2];
    ধ্রুবক পূর্ণসংখ্যা n_kv_heads = (পূর্ণসংখ্যা)K->ne[2];
    ধ্রুবক পূর্ণসংখ্যা n_batch    = (পূর্ণসংখ্যা)Q->ne[3];
    ধ্রুবক পূর্ণসংখ্যা n_kv       = (পূর্ণসংখ্যা)K->ne[1];
    ধ্রুবক পূর্ণসংখ্যা gqa_ratio  = n_q_heads / n_kv_heads;
    ধ্রুবক পূর্ণসংখ্যা n_query_rows = n_queries * gqa_ratio;

    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    GGML_ASSERT(max_bias == 0.0f);  // ALiBi not supported
    GGML_ASSERT(Q->ne[3] == K->ne[3] || K->ne[3] == 1);

    ধ্রুবক পূর্ণসংখ্যা chunk_আকার = std::min(MKL_FA_CHUNK_আকার_KV, n_kv);

    // Query rows are processed in tiles of q_tile_rows so the score buffers
    // (KQ_f32/S_f16 = q_tile_rows * chunk_size) stay bounded regardless of
    // batch size. n_query_rows <= Q_TILE is a single tile (no extra work).
    স্থির পূর্ণসংখ্যা q_tile_env = ggml_sycl_get_env("GGML_SYCL_MKL_FA_Q_TILE", MKL_FA_Q_TILE);
    ধ্রুবক পূর্ণসংখ্যা q_tile_rows = std::max(1, std::min(q_tile_env, n_query_rows));

    ধ্রুবক int64_t wg_আকার = MKL_FA_WG_আকার;

    // --- Debug output (gated by GGML_SYCL_MKL_FA_DEBUG=1) ---
    স্থির পূর্ণসংখ্যা mkl_call_গুনতি = 0;
    mkl_call_গুনতি++;
    স্থির পূর্ণসংখ্যা mkl_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    ধ্রুবক যৌক্তিক do_print = (mkl_debug == 1);

    ধ্রুবক int64_t q_row_stride  = Q->nb[1] / আকার(দশমিক);
    ধ্রুবক int64_t q_শীর্ষ_stride = Q->nb[2] / আকার(দশমিক);

    // Alias the dequantized buffers only when K and V expose the same values.
    // MLA V is a narrower view of K and needs its own strided dequantization.
    ধ্রুবক যৌক্তিক V_is_K_view = V->ne[0] == K->ne[0] && V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    // Early interleaved detection for debug output.
    // True interleaved detection happens after dequant (nb12_fp16 == nb11_fp16),
    // but we can pre-detect on the original tensor strides.
    ধ্রুবক যৌক্তিক k_early_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]);
    ধ্রুবক যৌক্তিক v_early_interleaved =
        !V_is_K_view && ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]);

    যদি (do_print) {
        GGML_LOG_INFO("[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d buf=%.1fMB%s%s\n",
                mkl_call_গুনতি, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_ratio, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_আকার,
                (দ্বিগুণ)((int64_t)n_query_rows * chunk_আকার * আকার(দশমিক))
                    / (1024.0 * 1024.0),
                k_early_interleaved ? " K_ILV" : "",
                v_early_interleaved ? " V_ILV" : "");
        GGML_LOG_INFO("[MKL-FA] #%d Q-nb1=%lld Q-nb2=%lld "
                "q_rs=%lld q_hs=%lld dst_rs=%lld dst_hs=%lld\n",
                mkl_call_গুনতি,
                (long long)Q->nb[1], (long long)Q->nb[2],
                (long long)q_row_stride, (long long)q_শীর্ষ_stride,
                (long long)(KQV->nb[1] / আকার(দশমিক)),
                (long long)(KQV->nb[2] / আকার(দশমিক)));
    }

    // --- Stream and allocators ---
    dpct::queue_ptr stream = ctx.stream();

#define MKL_TAKE_TIME(t0)  auto t0 = std::chrono::steady_clock::now()
#define MKL_ACCUM(acc, t0) করো { যদি (do_print) { \
    acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count(); \
} } যখন(0)

    int64_t gemm_kq_time_us  = 0;
    int64_t gemm_vkq_time_us = 0;
    int64_t softmax_time_us  = 0;
    int64_t dequant_time_us  = 0;

    MKL_TAKE_TIME(t_deq);

    // --- K/V dequant descriptors ---
    // Dequant is done per-chunk inside the KV loop (footprint independent of
    // context). Output is always dense row-major fp16 [this_chunk x D], lda=D.
    // Interleaved detection: ne[1]*nb[1] != nb[2] means heads are interleaved.
    ধ্রুবক যৌক্তিক k_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
    ধ্রুবক যৌক্তিক v_interleaved =
        ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

    ধ্রুবক mkl_fa_kv_desc K_desc = mkl_fa_make_desc(K, k_interleaved, n_kv_heads);
    ধ্রুবক mkl_fa_kv_desc V_desc = V_is_K_view
        ? K_desc : mkl_fa_make_desc(V, v_interleaved, n_kv_heads);

    MKL_ACCUM(dequant_time_us, t_deq);

    // --- Resolve mask pointers ---
    int64_t mask_শীর্ষ_stride = 0;
    int64_t mask_row_stride  = 0;
    পূর্ণসংখ্যা     mask_n_heads     = 0;

    যদি (mask) {
        // Use actual fp16 device size (2 bytes), NOT sizeof(sycl::half)
        // which may be 4 on the host in oneAPI.
        mask_শীর্ষ_stride = mask->nb[2] / 2;
        mask_row_stride  = mask->nb[1] / 2;
        mask_n_heads     = (পূর্ণসংখ্যা)mask->ne[2];
    }

    // --- Allocate intermediates from pool ---
    ggml_sycl_pool & pool = ctx.pool();

    ggml_sycl_pool_alloc<দশমিক>      KQ_f32(pool);      // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);       // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<দশমিক>      VKQ_chunk(pool);   // [q_tile_rows x DV]
    ggml_sycl_pool_alloc<দশমিক>      VKQ_accum(pool);   // [n_query_rows x DV] (full)
    ggml_sycl_pool_alloc<দশমিক>      KQ_max(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<দশমিক>      KQ_যোগ(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<sycl::half> Q_শীর্ষ_f16(pool);  // [n_query_rows x DKQ] (full)
    ggml_sycl_pool_alloc<sycl::half> K_chunk_f16(pool); // [chunk x DKQ] (per-chunk dequant)
    ggml_sycl_pool_alloc<sycl::half> V_chunk_f16(pool); // [chunk x DV] (per-chunk dequant)

    KQ_f32.alloc((size_t)q_tile_rows * chunk_আকার);
    S_f16.alloc((size_t)q_tile_rows * chunk_আকার);
    VKQ_chunk.alloc((size_t)q_tile_rows * DV);
    VKQ_accum.alloc((size_t)n_query_rows * DV);
    KQ_max.alloc(n_query_rows);
    KQ_যোগ.alloc(n_query_rows);
    Q_শীর্ষ_f16.alloc((size_t)n_query_rows * DKQ);
    K_chunk_f16.alloc((size_t)chunk_আকার * DKQ);

    sycl::half * V_chunk_f16_ptr;
    যদি (V_is_K_view) {
        V_chunk_f16_ptr = K_chunk_f16.ptr;   // V aliases K (DV == DKQ)
    } অন্যথা {
        V_chunk_f16.alloc((size_t)chunk_আকার * DV);
        V_chunk_f16_ptr = V_chunk_f16.ptr;
    }

    sycl::half * Q_শীর্ষ_f16_ptr  = Q_শীর্ষ_f16.ptr;
    দশমিক      * KQ_f32_ptr      = KQ_f32.ptr;
    sycl::half * S_f16_ptr       = S_f16.ptr;
    দশমিক      * VKQ_chunk_ptr   = VKQ_chunk.ptr;
    দশমিক      * VKQ_accum_ptr   = VKQ_accum.ptr;
    দশমিক      * KQ_max_ptr      = KQ_max.ptr;
    দশমিক      * KQ_যোগ_ptr      = KQ_যোগ.ptr;
    sycl::half * K_chunk_f16_ptr = K_chunk_f16.ptr;

    ধ্রুবক দশমিক alpha = 1.0f;
    ধ্রুবক দশমিক beta  = 0.0f;

    জন্য (পূর্ণসংখ্যা ib = 0; ib < n_batch; ib++) {
        ধ্রুবক দশমিক * Q_batch = (ধ্রুবক দশমিক *)Q->তথ্য
            + ib * (Q->nb[3] / আকার(দশমিক));
        দশমিক * dst_batch = (দশমিক *)KQV->তথ্য
            + ib * (KQV->nb[3] / আকার(দশমিক));

        ধ্রুবক sycl::half * mask_batch = nullptr;
        যদি (mask) {
            পূর্ণসংখ্যা m_batch = (mask->ne[3] > 1) ? ib : 0;
            mask_batch = (ধ্রুবক sycl::half *)mask->তথ্য
                + m_batch * (mask->nb[3] / 2);  // 2 = actual fp16 device size
        }

        জন্য (পূর্ণসংখ্যা ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            পূর্ণসংখ্যা kvh_base_শীর্ষ = ikvh * gqa_ratio;

            // 1. Pack all GQA Q heads into fp16 (full n_query_rows)
            mkl_fa_pack_q_fp16(stream,
                Q_শীর্ষ_f16_ptr, Q_batch,
                n_queries, DKQ,
                gqa_ratio, kvh_base_শীর্ষ,
                q_scale, q_row_stride, q_শীর্ষ_stride, wg_আকার);

            // 2. Initialize softmax state (full n_query_rows)
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_যোগ_ptr, VKQ_accum_ptr,
                n_query_rows, DV, wg_আকার);

            // Sync before MKL GEMM (MKL may use an internal queue)
            stream->wait();

            // 3. KV chunk loop (OUTER): dequant each chunk once, then tile queries.
            জন্য (পূর্ণসংখ্যা chunk_start = 0; chunk_start < n_kv; chunk_start += chunk_আকার) {
                পূর্ণসংখ্যা this_chunk = std::min(chunk_আকার, n_kv - chunk_start);

                // 3a. Dequant this KV chunk to dense fp16 (once per chunk)
                {
                    MKL_TAKE_TIME(t0);
                    mkl_fa_dequant_chunk(stream, K_desc, KQV,
                        K_chunk_f16_ptr, ikvh, chunk_start, this_chunk);
                    যদি (!V_is_K_view) {
                        mkl_fa_dequant_chunk(stream, V_desc, KQV,
                            V_chunk_f16_ptr, ikvh, chunk_start, this_chunk);
                    }
                    stream->wait();  // dequant must be ready before MKL GEMM
                    MKL_ACCUM(dequant_time_us, t0);
                }

                // 3b. Query tile loop (INNER) — bounds KQ_f32/S_f16 footprint.
                জন্য (পূর্ণসংখ্যা q0 = 0; q0 < n_query_rows; q0 += q_tile_rows) {
                    পূর্ণসংখ্যা q_rows = std::min(q_tile_rows, n_query_rows - q0);

                    // GEMM: KQ = Q_tile × K_chunk^T
                    {
                        MKL_TAKE_TIME(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::trans, transpose::nontrans,
                            this_chunk, q_rows, DKQ,
                            alpha,
                            K_chunk_f16_ptr, DKQ,
                            Q_শীর্ষ_f16_ptr + (int64_t)q0 * DKQ, DKQ,
                            beta,
                            KQ_f32_ptr, this_chunk);
                        চেষ্টা { ev.wait_and_throw(); } ধরো (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM KQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM KQ failed");
                        }
                        MKL_ACCUM(gemm_kq_time_us, t0);
                    }
                    // Online softmax over this chunk for this query tile
                    {
                        MKL_TAKE_TIME(t0);
                        mkl_fa_online_softmax_chunk(stream,
                            KQ_f32_ptr, S_f16_ptr,
                            KQ_max_ptr, KQ_যোগ_ptr, VKQ_accum_ptr,
                            q0, q_rows, n_queries, DV,
                            this_chunk, chunk_start,
                            kvh_base_শীর্ষ,
                            mask_batch, mask_শীর্ষ_stride,
                            mask_row_stride, mask_n_heads,
                            logit_softcap, wg_আকার);
                        stream->wait();  // S_f16 must be ready for GEMM
                        MKL_ACCUM(softmax_time_us, t0);
                    }

                    // GEMM: VKQ_chunk = S × V_chunk
                    {
                        MKL_TAKE_TIME(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::nontrans, transpose::nontrans,
                            DV, q_rows, this_chunk,
                            alpha,
                            V_chunk_f16_ptr, DV,
                            S_f16_ptr, this_chunk,
                            beta,
                            VKQ_chunk_ptr, DV);
                        চেষ্টা { ev.wait_and_throw(); } ধরো (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM VKQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM VKQ failed");
                        }
                        MKL_ACCUM(gemm_vkq_time_us, t0);
                    }
                    // VKQ_accum[q0..] += VKQ_chunk
                    {
                        ধ্রুবক int64_t n_যোগ = (int64_t)q_rows * DV;
                        ধ্রুবক int64_t wg = ((n_যোগ + wg_আকার - 1) / wg_আকার)
                            * wg_আকার;
                        দশমিক * accum = VKQ_accum_ptr + (int64_t)q0 * DV;
                        stream->submit([&](sycl::handler & cgh) {
                            cgh.parallel_for(sycl::nd_range<1>(wg, wg_আকার),
                                [=](sycl::nd_item<1> item) {
                                    int64_t i = item.get_global_id(0);
                                    যদি (i < n_যোগ) {
                                        accum[i] += VKQ_chunk_ptr[i];
                                    }
                                });
                        });
                    }
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            জন্য (পূর্ণসংখ্যা iqg = 0; iqg < gqa_ratio; iqg++) {
                পূর্ণসংখ্যা     iqh        = kvh_base_শীর্ষ + iqg;
                int64_t src_offset = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_শীর্ষ(stream,
                    dst_batch, VKQ_accum_ptr, KQ_যোগ_ptr,
                    iqh, n_queries, DV, n_q_heads,
                    src_offset, wg_আকার);
            }
        }
    }

#undef MKL_TAKE_TIME
#undef MKL_ACCUM

    যদি (do_print) {
        ধ্রুবক int64_t v_chunk_elems = V_is_K_view ? 0 : (int64_t)chunk_আকার * DV;
        দ্বিগুণ যোগ_mb = (দ্বিগুণ)(
            (int64_t)q_tile_rows * chunk_আকার * আকার(দশমিক)      // KQ_f32
          + (int64_t)q_tile_rows * chunk_আকার * আকার(sycl::half) // S_f16
          + (int64_t)q_tile_rows * DV * আকার(দশমিক)              // VKQ_chunk
          + (int64_t)n_query_rows * DV * আকার(দশমিক)             // VKQ_accum
          + (int64_t)n_query_rows * আকার(দশমিক)                  // KQ_max
          + (int64_t)n_query_rows * আকার(দশমিক)                  // KQ_sum
          + (int64_t)n_query_rows * DKQ * আকার(sycl::half)       // Q_head_f16
          + (int64_t)chunk_আকার * DKQ * আকার(sycl::half)         // K_chunk_f16
          + v_chunk_elems * (int64_t)আকার(sycl::half)            // V_chunk_f16
        ) / (1024.0 * 1024.0);
        GGML_LOG_INFO("[MKL-FA] #%d n_kv=%d n_q=%d q_tile=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld "
                "buf_mb=%.1f\n",
                mkl_call_গুনতি, n_kv, n_queries, q_tile_rows,
                (long long)dequant_time_us,
                (long long)gemm_kq_time_us,
                (long long)softmax_time_us,
                (long long)gemm_vkq_time_us,
                যোগ_mb);
    }
}
