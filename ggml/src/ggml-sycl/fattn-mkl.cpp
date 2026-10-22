// Flash attention via oneMKL GEMM (XMX-accelerated).
// Uses column_major::gemm for Q*K^T and S*V matmuls
// with an online softmax SYCL kernel.
//
// All GQA query heads sharing a KV head are batched into single
// GEMM calls, amortizing MKL launch overhead across K and V reuse.
//

#অন্তর্ভাৱ "common.hpp"
#অন্তর্ভাৱ "fattn-common.hpp"
#অন্তর্ভাৱ "convert.hpp"
#অন্তর্ভাৱ "fattn.hpp"

#অন্তর্ভাৱ <oneapi/mkl.hpp>
#অন্তর্ভাৱ <cstdio>
#অন্তর্ভাৱ <chrono>

#define MKL_FA_CHUNK_SIZE_KV 8192

// Number of query rows processed per tile. The score buffers (KQ_f32, S_f16)
// are sized q_tile_rows * chunk_size, so this bounds their footprint
// regardless of batch size (n_query_rows = n_queries * gqa_ratio). A typical
// single-ubatch prefill (e.g. ubatch 1024 * gqa 8 = 8192 rows) is exactly one
// tile, so it runs with no extra iterations. Larger batches tile and stay
// bounded. Override with GGML_SYCL_MKL_FA_Q_TILE.
#define MKL_FA_Q_TILE 8192

#define MKL_FA_WG_SIZE 256

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
    ধ্রুৱ ভিন্ন * __restrict q_src,
    পূর্ণাঙ্ক n_queries, পূর্ণাঙ্ক DKQ,
    পূর্ণাঙ্ক gqa_ratio, পূর্ণাঙ্ক kvh_base_head,
    ভিন্ন q_scale, int64_t q_row_stride, int64_t q_head_stride,
    int64_t wg_size) {

    বাবে (পূর্ণাঙ্ক iqg = 0; iqg < gqa_ratio; iqg++) {
        পূর্ণাঙ্ক     iqh       = kvh_base_head + iqg;
        sycl::half * dst_g = dst + (int64_t)iqg * n_queries * DKQ;

        ধ্রুৱ int64_t n_elem = (int64_t)n_queries * DKQ;
        ধ্রুৱ int64_t wg = ((n_elem + wg_size - 1) / wg_size) * wg_size;

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
                [=](sycl::nd_item<1> item) {
                    int64_t e = item.get_global_id(0);
                    যদি (e >= n_elem) ঘূৰাই;

                    int64_t q = e / DKQ;
                    int64_t d = e - q * DKQ;

                    // Stride-aware source offset: handles permuted,
                    // sliced, or contiguous Q tensor layouts.
                    int64_t src_off = d
                        + q * q_row_stride
                        + (int64_t)iqh * q_head_stride;

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
    ভিন্ন * kmax, ভিন্ন * ksum, ভিন্ন * vacc,
    পূর্ণাঙ্ক n_query_rows, পূর্ণাঙ্ক DV, int64_t wg_size) {

    ধ্রুৱ ভিন্ন    neg_inf   = -1e30f;
    ধ্রুৱ int64_t  n_maxsum  = n_query_rows;
    ধ্রুৱ int64_t  n_vacc    = (int64_t)n_query_rows * DV;
    ধ্রুৱ int64_t  total     = (n_vacc > n_maxsum) ? n_vacc : n_maxsum;
    ধ্রুৱ int64_t  wg = ((total + wg_size - 1) / wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
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
    ভিন্ন * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    ভিন্ন * __restrict KQ_max,
    ভিন্ন * __restrict KQ_sum,
    ভিন্ন * __restrict VKQ_accum,
    পূর্ণাঙ্ক q0, পূর্ণাঙ্ক q_rows, পূর্ণাঙ্ক n_queries, পূর্ণাঙ্ক DV,
    পূর্ণাঙ্ক chunk_size, পূর্ণাঙ্ক chunk_start,
    পূর্ণাঙ্ক kvh_head,
    ধ্রুৱ sycl::half * mask_data, int64_t mask_head_stride,
    int64_t mask_row_stride, পূর্ণাঙ্ক mask_n_heads,
    ভিন্ন logit_softcap, int64_t wg_size) {

    // One work-group per query row: exactly q_rows groups of wg_size
    // items. q_rows * wg_size is already a multiple of wg_size, so unlike
    // the one-item-per-row kernels there is no round-up / tail guard.
    ধ্রুৱ int64_t wg         = q_rows * wg_size;
    ধ্রুৱ পূর্ণাঙ্ক     local_size = (পূর্ণাঙ্ক) wg_size;  // stride in the loops below
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
            [=](sycl::nd_item<1> item) {
                ধ্রুৱ পূর্ণাঙ্ক local_id = (পূর্ণাঙ্ক)item.get_local_id(0);
                ধ্রুৱ পূর্ণাঙ্ক row = (পূর্ণাঙ্ক)item.get_group(0); // tile-relative
                ধ্রুৱ পূর্ণাঙ্ক jc_abs    = q0 + row;
                ধ্রুৱ পূর্ণাঙ্ক gqa_group = jc_abs / n_queries;
                ধ্রুৱ পূর্ণাঙ্ক q_row     = jc_abs % n_queries;
                // Score buffers are tile-local (relative index).
                ধ্রুৱ ভিন্ন * __restrict KQ_row = KQ_f32
                    + row * (int64_t)chunk_size;
                sycl::half * __restrict S_row = S_f16
                    + row * (int64_t)chunk_size;
                // Persistent accumulator is full-sized (absolute index).
                ভিন্ন * __restrict vkq = VKQ_accum
                    + jc_abs * (int64_t)DV;
                ধ্রুৱ sycl::half * mask_h = nullptr;
                int64_t m_stride = 0;
                যদি (mask_data) {
                    পূর্ণাঙ্ক m_head = (mask_n_heads > 1)
                        ? (kvh_head + gqa_group) : 0;
                    mask_h   = mask_data + (int64_t)m_head * mask_head_stride;
                    m_stride = mask_row_stride;
                }
                // Score at chunk offset i — original per-element math.
                auto score = [&](পূর্ণাঙ্ক i) {
                    ভিন্ন s = KQ_row[i];
                    যদি (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    যদি (mask_h) {
                        s += (ভিন্ন)mask_h[q_row * m_stride
                            + (chunk_start + i)];
                    }
                    ঘূৰাই s;
                };
                // Pass 1: strided (coalesced) row-wise local maximum.
                ভিন্ন local_max = -1e30f;
                বাবে (পূর্ণাঙ্ক i = local_id; i < chunk_size; i += local_size) {
                    ভিন্ন s = score(i);
                    যদি (s > local_max) local_max = s;
                }
                ধ্রুৱ ভিন্ন final_local_max = sycl::reduce_over_group(
                    item.get_group(), local_max, sycl::maximum<ভিন্ন>());
                // Rescale previous accumulator by exp(old_max - new_max)
                ভিন্ন old_max = KQ_max[jc_abs];
                ভিন্ন new_max = (old_max > final_local_max) ? old_max : final_local_max;
                ভিন্ন rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::exp(old_max - new_max);
                বাবে (পূর্ণাঙ্ক v = local_id; v < DV; v += local_size) {
                    vkq[v] *= rescale;
                }
                // Pass 2: softmax numerators, strided; S row written once.
                ভিন্ন local_sum = 0.0f;
                বাবে (পূর্ণাঙ্ক i = local_id; i < chunk_size; i += local_size) {
                    ভিন্ন s = score(i);
                    ভিন্ন val = sycl::native::exp(s - new_max);
                    S_row[i] = sycl::half(val);
                    local_sum += val;
                }
                ধ্রুৱ ভিন্ন total_sum = sycl::reduce_over_group(
                    item.get_group(), local_sum, sycl::plus<ভিন্ন>());
                যদি (local_id == 0) {
                    KQ_sum[jc_abs] = KQ_sum[jc_abs] * rescale + total_sum;
                    KQ_max[jc_abs] = new_max;
                }
            });
    });
}

// Write one GQA group's normalized output to its destination head.
স্থির শূন্য mkl_fa_normalize_head(
    dpct::queue_ptr stream,
    ভিন্ন * __restrict dst_batch,
    ধ্রুৱ ভিন্ন * __restrict VKQ_accum,
    ধ্রুৱ ভিন্ন * __restrict KQ_sum,
    পূর্ণাঙ্ক iqh, পূর্ণাঙ্ক n_queries, পূর্ণাঙ্ক DV, পূর্ণাঙ্ক n_q_heads,
    int64_t src_offset, int64_t wg_size) {

    ধ্রুৱ int64_t wg = ((n_queries + wg_size - 1) / wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
            [=](sycl::nd_item<1> item) {
                পূর্ণাঙ্ক jc = item.get_global_id(0);
                যদি (jc >= n_queries) ঘূৰাই;

                পূর্ণাঙ্ক     ksum_idx = (পূর্ণাঙ্ক)(src_offset / DV) + jc;
                ভিন্ন   inv_sum  = 1.0f / KQ_sum[ksum_idx];
                ধ্রুৱ ভিন্ন * __restrict src = VKQ_accum
                    + src_offset + jc * (int64_t)DV;
                // Interleaved dst layout (matching TILE):
                // rows alternate between heads, then increment query.
                // offset = (query * n_q_heads + head) * DV
                ভিন্ন * __restrict dst_row = dst_batch
                    + ((int64_t)jc * n_q_heads + iqh) * (int64_t)DV;

                বাবে (পূর্ণাঙ্ক v = 0; v < DV; v++) {
                    dst_row[v] = src[v] * inv_sum;
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

সংরচনা mkl_fa_kv_desc {
    ধ্রুৱ ৱর্ণ *         data = nullptr;
    ggml_type            type = GGML_TYPE_F16;
    int64_t              D    = 0;      // ne[0]
    int64_t              nb1  = 0;      // byte stride, seq dim
    int64_t              nb2  = 0;      // byte stride, head dim
    mkl_fa_kv_desc_mode  mode = MKL_FA_KV_MODE_F16_DENSE;
    int64_t              ts   = 0;      // type size (mode 3 base offset)
    int64_t              s01  = 0;      // nc row stride in blocks (mode 3)
    int64_t              s02  = 0;      // nc head stride in blocks (mode 3)
};

স্থির mkl_fa_kv_desc mkl_fa_make_desc(ধ্রুৱ ggml_tensor * T, তার্কিক interleaved, পূর্ণাঙ্ক n_kv_heads) {
    mkl_fa_kv_desc d;
    d.data = (ধ্রুৱ ৱর্ণ *)T->data;
    d.type = T->type;
    d.D    = T->ne[0];
    d.nb1  = (int64_t)T->nb[1];
    d.nb2  = (int64_t)T->nb[2];
    d.ts   = (int64_t)ggml_type_size(T->type);

    যদি (T->type == GGML_TYPE_F16) {
        // MLA's V cache is a 512-wide view of 576-wide K rows. Treat any
        // padded row stride as strided even when there is only one KV head.
        d.mode = interleaved || d.nb1 != d.D * (int64_t)পরিমাণ(sycl::half)
            ? MKL_FA_KV_MODE_F16_INTERLEAVED : MKL_FA_KV_MODE_F16_DENSE;
    } অন্যথা যদি (ggml_is_contiguously_allocated(T) && !interleaved) {
        d.mode = MKL_FA_KV_MODE_QUANT_CONTIG;
    } অন্যথা {
        d.mode = MKL_FA_KV_MODE_QUANT_NC;
        ধ্রুৱ int64_t bs          = (int64_t)ggml_blck_size(T->type);
        ধ্রুৱ int64_t blk_per_row = T->ne[0] / bs;
        // True Gemma interleave packs heads within a row (nb[2] < ne[1]*nb[1])
        // → reconstruct physical strides. Padded seq-views (nb[2] > ne[1]*nb[1])
        // already have correct physical strides.
        ধ্রুৱ তার্কিক gemma = interleaved &&
            ((int64_t)T->nb[2] < (int64_t)T->ne[1] * (int64_t)T->nb[1]);
        যদি (gemma) {
            d.s01 = (int64_t)n_kv_heads * blk_per_row;
            d.s02 = blk_per_row;
        } অন্যথা {
            d.s01 = d.nb1 / d.ts;
            d.s02 = d.nb2 / d.ts;
        }
    }
    ঘূৰাই d;
}

// Dequant one KV-head chunk into a dense [this_chunk x D] fp16 buffer.
স্থির শূন্য mkl_fa_dequant_chunk(
    dpct::queue_ptr stream, ধ্রুৱ mkl_fa_kv_desc & d, ggml_tensor * dst_ctx,
    sycl::half * out, পূর্ণাঙ্ক ikvh, পূর্ণাঙ্ক chunk_start, পূর্ণাঙ্ক this_chunk) {

    ধ্রুৱ int64_t D = d.D;
    ৱিকল্প (d.mode) {
        পক্ষ MKL_FA_KV_MODE_F16_DENSE: {
            ধ্রুৱ ৱর্ণ * base = d.data + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            stream->memcpy(out, base, (size_t)this_chunk * D * পরিমাণ(sycl::half));
            ভঙ্গ;
        }
        পক্ষ MKL_FA_KV_MODE_F16_INTERLEAVED: {
            ধ্রুৱ ৱর্ণ * base = d.data + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            ধ্রুৱ int64_t row_halfs = d.nb1 / (int64_t)পরিমাণ(sycl::half);
            ধ্রুৱ sycl::half * src = (ধ্রুৱ sycl::half *)base;
            stream->parallel_for(
                sycl::range<2>((size_t)this_chunk, (size_t)D),
                [=](sycl::item<2> it) {
                    int64_t r = it.get_id(0);
                    int64_t c = it.get_id(1);
                    out[r * D + c] = src[r * row_halfs + c];
                });
            ভঙ্গ;
        }
        পক্ষ MKL_FA_KV_MODE_QUANT_CONTIG: {
            ধ্রুৱ ৱর্ণ * base = d.data + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(d.type, dst_ctx);
            to_fp16(base, out, (int64_t)this_chunk * D, stream);
            ভঙ্গ;
        }
        স্ৱভাৱ: {  // MKL_FA_KV_MODE_QUANT_NC
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(d.type);
            ধ্রুৱ int64_t base_blocks = (int64_t)ikvh * d.s02
                + (int64_t)chunk_start * d.s01;
            ধ্রুৱ ৱর্ণ * base = d.data + base_blocks * d.ts;
            // ne02 = ne03 = 1 → s02/s03 inert; head+chunk offset carried by base.
            to_fp16(base, out, D, this_chunk, 1, 1, d.s01, d.s02, d.s02, stream);
            ভঙ্গ;
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

    ধ্রুৱ ggml_tensor * Q    = dst->src[0];
    ধ্রুৱ ggml_tensor * K    = dst->src[1];
    ধ্রুৱ ggml_tensor * V    = dst->src[2];
    ধ্রুৱ ggml_tensor * mask = dst->src[3];
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    ভিন্ন scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (ধ্রুৱ ভিন্ন *)KQV->op_params + 0, পরিমাণ(ভিন্ন));
    memcpy(&max_bias,      (ধ্রুৱ ভিন্ন *)KQV->op_params + 1, পরিমাণ(ভিন্ন));
    memcpy(&logit_softcap, (ধ্রুৱ ভিন্ন *)KQV->op_params + 2, পরিমাণ(ভিন্ন));

    ধ্রুৱ ভিন্ন q_scale = scale;

    // --- Dimensions ---
    ধ্রুৱ পূর্ণাঙ্ক DKQ        = (পূর্ণাঙ্ক)K->ne[0];
    ধ্রুৱ পূর্ণাঙ্ক DV         = (পূর্ণাঙ্ক)V->ne[0];
    ধ্রুৱ পূর্ণাঙ্ক n_queries  = (পূর্ণাঙ্ক)Q->ne[1];
    ধ্রুৱ পূর্ণাঙ্ক n_q_heads  = (পূর্ণাঙ্ক)Q->ne[2];
    ধ্রুৱ পূর্ণাঙ্ক n_kv_heads = (পূর্ণাঙ্ক)K->ne[2];
    ধ্রুৱ পূর্ণাঙ্ক n_batch    = (পূর্ণাঙ্ক)Q->ne[3];
    ধ্রুৱ পূর্ণাঙ্ক n_kv       = (পূর্ণাঙ্ক)K->ne[1];
    ধ্রুৱ পূর্ণাঙ্ক gqa_ratio  = n_q_heads / n_kv_heads;
    ধ্রুৱ পূর্ণাঙ্ক n_query_rows = n_queries * gqa_ratio;

    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    GGML_ASSERT(max_bias == 0.0f);  // ALiBi not supported
    GGML_ASSERT(Q->ne[3] == K->ne[3] || K->ne[3] == 1);

    ধ্রুৱ পূর্ণাঙ্ক chunk_size = std::min(MKL_FA_CHUNK_SIZE_KV, n_kv);

    // Query rows are processed in tiles of q_tile_rows so the score buffers
    // (KQ_f32/S_f16 = q_tile_rows * chunk_size) stay bounded regardless of
    // batch size. n_query_rows <= Q_TILE is a single tile (no extra work).
    স্থির পূর্ণাঙ্ক q_tile_env = ggml_sycl_get_env("GGML_SYCL_MKL_FA_Q_TILE", MKL_FA_Q_TILE);
    ধ্রুৱ পূর্ণাঙ্ক q_tile_rows = std::max(1, std::min(q_tile_env, n_query_rows));

    ধ্রুৱ int64_t wg_size = MKL_FA_WG_SIZE;

    // --- Debug output (gated by GGML_SYCL_MKL_FA_DEBUG=1) ---
    স্থির পূর্ণাঙ্ক mkl_call_গুনতি = 0;
    mkl_call_গুনতি++;
    স্থির পূর্ণাঙ্ক mkl_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    ধ্রুৱ তার্কিক do_print = (mkl_debug == 1);

    ধ্রুৱ int64_t q_row_stride  = Q->nb[1] / পরিমাণ(ভিন্ন);
    ধ্রুৱ int64_t q_head_stride = Q->nb[2] / পরিমাণ(ভিন্ন);

    // Alias the dequantized buffers only when K and V expose the same values.
    // MLA V is a narrower view of K and needs its own strided dequantization.
    ধ্রুৱ তার্কিক V_is_K_view = V->ne[0] == K->ne[0] && V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    // Early interleaved detection for debug output.
    // True interleaved detection happens after dequant (nb12_fp16 == nb11_fp16),
    // but we can pre-detect on the original tensor strides.
    ধ্রুৱ তার্কিক k_early_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]);
    ধ্রুৱ তার্কিক v_early_interleaved =
        !V_is_K_view && ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]);

    যদি (do_print) {
        GGML_LOG_INFO("[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d buf=%.1fMB%s%s\n",
                mkl_call_গুনতি, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_ratio, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_size,
                (দ্ৱিগুণ)((int64_t)n_query_rows * chunk_size * পরিমাণ(ভিন্ন))
                    / (1024.0 * 1024.0),
                k_early_interleaved ? " K_ILV" : "",
                v_early_interleaved ? " V_ILV" : "");
        GGML_LOG_INFO("[MKL-FA] #%d Q-nb1=%lld Q-nb2=%lld "
                "q_rs=%lld q_hs=%lld dst_rs=%lld dst_hs=%lld\n",
                mkl_call_গুনতি,
                (long long)Q->nb[1], (long long)Q->nb[2],
                (long long)q_row_stride, (long long)q_head_stride,
                (long long)(KQV->nb[1] / পরিমাণ(ভিন্ন)),
                (long long)(KQV->nb[2] / পরিমাণ(ভিন্ন)));
    }

    // --- Stream and allocators ---
    dpct::queue_ptr stream = ctx.stream();

#define MKL_TAKE_TIME(t0)  auto t0 = std::chrono::steady_clock::now()
#define MKL_ACCUM(acc, t0) কৰা { যদি (do_print) { \
    acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count(); \
} } যেতিয়ালৈ(0)

    int64_t gemm_kq_time_us  = 0;
    int64_t gemm_vkq_time_us = 0;
    int64_t softmax_time_us  = 0;
    int64_t dequant_time_us  = 0;

    MKL_TAKE_TIME(t_deq);

    // --- K/V dequant descriptors ---
    // Dequant is done per-chunk inside the KV loop (footprint independent of
    // context). Output is always dense row-major fp16 [this_chunk x D], lda=D.
    // Interleaved detection: ne[1]*nb[1] != nb[2] means heads are interleaved.
    ধ্রুৱ তার্কিক k_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
    ধ্রুৱ তার্কিক v_interleaved =
        ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

    ধ্রুৱ mkl_fa_kv_desc K_desc = mkl_fa_make_desc(K, k_interleaved, n_kv_heads);
    ধ্রুৱ mkl_fa_kv_desc V_desc = V_is_K_view
        ? K_desc : mkl_fa_make_desc(V, v_interleaved, n_kv_heads);

    MKL_ACCUM(dequant_time_us, t_deq);

    // --- Resolve mask pointers ---
    int64_t mask_head_stride = 0;
    int64_t mask_row_stride  = 0;
    পূর্ণাঙ্ক     mask_n_heads     = 0;

    যদি (mask) {
        // Use actual fp16 device size (2 bytes), NOT sizeof(sycl::half)
        // which may be 4 on the host in oneAPI.
        mask_head_stride = mask->nb[2] / 2;
        mask_row_stride  = mask->nb[1] / 2;
        mask_n_heads     = (পূর্ণাঙ্ক)mask->ne[2];
    }

    // --- Allocate intermediates from pool ---
    ggml_sycl_pool & pool = ctx.pool();

    ggml_sycl_pool_alloc<ভিন্ন>      KQ_f32(pool);      // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);       // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<ভিন্ন>      VKQ_chunk(pool);   // [q_tile_rows x DV]
    ggml_sycl_pool_alloc<ভিন্ন>      VKQ_accum(pool);   // [n_query_rows x DV] (full)
    ggml_sycl_pool_alloc<ভিন্ন>      KQ_max(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<ভিন্ন>      KQ_sum(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<sycl::half> Q_head_f16(pool);  // [n_query_rows x DKQ] (full)
    ggml_sycl_pool_alloc<sycl::half> K_chunk_f16(pool); // [chunk x DKQ] (per-chunk dequant)
    ggml_sycl_pool_alloc<sycl::half> V_chunk_f16(pool); // [chunk x DV] (per-chunk dequant)

    KQ_f32.alloc((size_t)q_tile_rows * chunk_size);
    S_f16.alloc((size_t)q_tile_rows * chunk_size);
    VKQ_chunk.alloc((size_t)q_tile_rows * DV);
    VKQ_accum.alloc((size_t)n_query_rows * DV);
    KQ_max.alloc(n_query_rows);
    KQ_sum.alloc(n_query_rows);
    Q_head_f16.alloc((size_t)n_query_rows * DKQ);
    K_chunk_f16.alloc((size_t)chunk_size * DKQ);

    sycl::half * V_chunk_f16_ptr;
    যদি (V_is_K_view) {
        V_chunk_f16_ptr = K_chunk_f16.ptr;   // V aliases K (DV == DKQ)
    } অন্যথা {
        V_chunk_f16.alloc((size_t)chunk_size * DV);
        V_chunk_f16_ptr = V_chunk_f16.ptr;
    }

    sycl::half * Q_head_f16_ptr  = Q_head_f16.ptr;
    ভিন্ন      * KQ_f32_ptr      = KQ_f32.ptr;
    sycl::half * S_f16_ptr       = S_f16.ptr;
    ভিন্ন      * VKQ_chunk_ptr   = VKQ_chunk.ptr;
    ভিন্ন      * VKQ_accum_ptr   = VKQ_accum.ptr;
    ভিন্ন      * KQ_max_ptr      = KQ_max.ptr;
    ভিন্ন      * KQ_sum_ptr      = KQ_sum.ptr;
    sycl::half * K_chunk_f16_ptr = K_chunk_f16.ptr;

    ধ্রুৱ ভিন্ন alpha = 1.0f;
    ধ্রুৱ ভিন্ন beta  = 0.0f;

    বাবে (পূর্ণাঙ্ক ib = 0; ib < n_batch; ib++) {
        ধ্রুৱ ভিন্ন * Q_batch = (ধ্রুৱ ভিন্ন *)Q->data
            + ib * (Q->nb[3] / পরিমাণ(ভিন্ন));
        ভিন্ন * dst_batch = (ভিন্ন *)KQV->data
            + ib * (KQV->nb[3] / পরিমাণ(ভিন্ন));

        ধ্রুৱ sycl::half * mask_batch = nullptr;
        যদি (mask) {
            পূর্ণাঙ্ক m_batch = (mask->ne[3] > 1) ? ib : 0;
            mask_batch = (ধ্রুৱ sycl::half *)mask->data
                + m_batch * (mask->nb[3] / 2);  // 2 = actual fp16 device size
        }

        বাবে (পূর্ণাঙ্ক ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            পূর্ণাঙ্ক kvh_base_head = ikvh * gqa_ratio;

            // 1. Pack all GQA Q heads into fp16 (full n_query_rows)
            mkl_fa_pack_q_fp16(stream,
                Q_head_f16_ptr, Q_batch,
                n_queries, DKQ,
                gqa_ratio, kvh_base_head,
                q_scale, q_row_stride, q_head_stride, wg_size);

            // 2. Initialize softmax state (full n_query_rows)
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_sum_ptr, VKQ_accum_ptr,
                n_query_rows, DV, wg_size);

            // Sync before MKL GEMM (MKL may use an internal queue)
            stream->wait();

            // 3. KV chunk loop (OUTER): dequant each chunk once, then tile queries.
            বাবে (পূর্ণাঙ্ক chunk_start = 0; chunk_start < n_kv; chunk_start += chunk_size) {
                পূর্ণাঙ্ক this_chunk = std::min(chunk_size, n_kv - chunk_start);

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
                বাবে (পূর্ণাঙ্ক q0 = 0; q0 < n_query_rows; q0 += q_tile_rows) {
                    পূর্ণাঙ্ক q_rows = std::min(q_tile_rows, n_query_rows - q0);

                    // GEMM: KQ = Q_tile × K_chunk^T
                    {
                        MKL_TAKE_TIME(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::trans, transpose::nontrans,
                            this_chunk, q_rows, DKQ,
                            alpha,
                            K_chunk_f16_ptr, DKQ,
                            Q_head_f16_ptr + (int64_t)q0 * DKQ, DKQ,
                            beta,
                            KQ_f32_ptr, this_chunk);
                        যত্ন { ev.wait_and_throw(); } গৃহাণ (sycl::exception & e) {
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
                            KQ_max_ptr, KQ_sum_ptr, VKQ_accum_ptr,
                            q0, q_rows, n_queries, DV,
                            this_chunk, chunk_start,
                            kvh_base_head,
                            mask_batch, mask_head_stride,
                            mask_row_stride, mask_n_heads,
                            logit_softcap, wg_size);
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
                        যত্ন { ev.wait_and_throw(); } গৃহাণ (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM VKQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM VKQ failed");
                        }
                        MKL_ACCUM(gemm_vkq_time_us, t0);
                    }
                    // VKQ_accum[q0..] += VKQ_chunk
                    {
                        ধ্রুৱ int64_t n_total = (int64_t)q_rows * DV;
                        ধ্রুৱ int64_t wg = ((n_total + wg_size - 1) / wg_size)
                            * wg_size;
                        ভিন্ন * accum = VKQ_accum_ptr + (int64_t)q0 * DV;
                        stream->submit([&](sycl::handler & cgh) {
                            cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
                                [=](sycl::nd_item<1> item) {
                                    int64_t i = item.get_global_id(0);
                                    যদি (i < n_total) {
                                        accum[i] += VKQ_chunk_ptr[i];
                                    }
                                });
                        });
                    }
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            বাবে (পূর্ণাঙ্ক iqg = 0; iqg < gqa_ratio; iqg++) {
                পূর্ণাঙ্ক     iqh        = kvh_base_head + iqg;
                int64_t src_offset = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_head(stream,
                    dst_batch, VKQ_accum_ptr, KQ_sum_ptr,
                    iqh, n_queries, DV, n_q_heads,
                    src_offset, wg_size);
            }
        }
    }

#undef MKL_TAKE_TIME
#undef MKL_ACCUM

    যদি (do_print) {
        ধ্রুৱ int64_t v_chunk_elems = V_is_K_view ? 0 : (int64_t)chunk_size * DV;
        দ্ৱিগুণ total_mb = (দ্ৱিগুণ)(
            (int64_t)q_tile_rows * chunk_size * পরিমাণ(ভিন্ন)      // KQ_f32
          + (int64_t)q_tile_rows * chunk_size * পরিমাণ(sycl::half) // S_f16
          + (int64_t)q_tile_rows * DV * পরিমাণ(ভিন্ন)              // VKQ_chunk
          + (int64_t)n_query_rows * DV * পরিমাণ(ভিন্ন)             // VKQ_accum
          + (int64_t)n_query_rows * পরিমাণ(ভিন্ন)                  // KQ_max
          + (int64_t)n_query_rows * পরিমাণ(ভিন্ন)                  // KQ_sum
          + (int64_t)n_query_rows * DKQ * পরিমাণ(sycl::half)       // Q_head_f16
          + (int64_t)chunk_size * DKQ * পরিমাণ(sycl::half)         // K_chunk_f16
          + v_chunk_elems * (int64_t)পরিমাণ(sycl::half)            // V_chunk_f16
        ) / (1024.0 * 1024.0);
        GGML_LOG_INFO("[MKL-FA] #%d n_kv=%d n_q=%d q_tile=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld "
                "buf_mb=%.1f\n",
                mkl_call_গুনতি, n_kv, n_queries, q_tile_rows,
                (long long)dequant_time_us,
                (long long)gemm_kq_time_us,
                (long long)softmax_time_us,
                (long long)gemm_vkq_time_us,
                total_mb);
    }
}
